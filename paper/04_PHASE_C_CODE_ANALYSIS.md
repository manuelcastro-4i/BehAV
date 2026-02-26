# 6. Phase (c) Deep Dive — Behavioral Conflict Analysis in BehAV Code

This section maps exactly WHERE and HOW behavioral conflicts manifest in the BehAV codebase, with line numbers, execution traces, and dialogue injection points.

## 6.1 Cost Map Conflict Mechanics

**Hardcoded CLIPSeg prompts** (`behav-planner.py:439-440`):
```python
self.prompts = ["vegetation", "Pavement","grass", "Stop gesture"]
```
These are never updated from the instruction decomposition module. The two modules are completely decoupled.

**Fixed cost values** (`behav-planner.py:442`):
```python
self.cost_values = [0.05, 0.95, 0.48, 0]  # high=preferred, low=avoid
```

| Index | Prompt | Cost | Meaning |
|-------|--------|------|---------|
| 0 | vegetation | 0.05 | Strongly avoid |
| 1 | Pavement | 0.95 | Strongly prefer |
| 2 | grass | 0.48 | Mildly avoid |
| 3 | Stop gesture | 0.00 | Maximally avoid |

**Sequential overwriting loop** (`behav-planner.py:826-837`):
```python
combined_cost_map = np.full((self.img_h, self.img_w), 128, dtype=np.float32)
preds_resized = F.interpolate(preds.unsqueeze(1), ...)
for i, pred_resized in enumerate(preds_resized):
    mask = pred_resized > 0.1
    combined_cost_map[mask] = pred_resized[mask] * 255 * self.cost_values[i]
```
Later prompts **silently overwrite** earlier ones at overlapping pixels. No blending, no max, no conflict detection. The resolution is entirely dependent on prompt ordering.

**MAX cost metric** (`behav-planner.py:908-910`):
```python
costs = self.behav_costmap[valid_y, valid_x]
max_cost = np.max(costs) if costs.size > 0 else 0.0
```
A single bad pixel along the trajectory determines its behavioral cost.

**No quality gate after optimization** (`behav-planner.py:543-556`):
The optimizer returns a trajectory and it is **always executed**, regardless of quality. No convergence check, no minimum threshold, no fallback.

**Out-of-frame escape hatch** (`behav-planner.py:900-906`):
```python
if len(valid_x) == 0:
    return marked_img, 0.0  # Zero cost = optimal under minimization
```
Trajectories projecting outside the camera get cost 0.0, creating a perverse incentive to steer off-camera.

## 6.2 Inverted Cost Semantics Bug

The comment on line 442 says "high means preferred," but the optimizer **minimizes** `total_cost` (line 616). The flow:

1. Pavement pixel: `0.8 * 255 * 0.95 = 193.8` (high value)
2. Vegetation pixel: `0.8 * 255 * 0.05 = 10.2` (low value)
3. `expected_behav = max_cost / 255` → Pavement = 0.76, Vegetation = 0.04
4. Optimizer minimizes `total_cost = expected_collision + expected_progress + expected_action + expected_behav`
5. **Result: optimizer AVOIDS pavement (0.76 penalty) and PREFERS vegetation (0.04 penalty)** — the opposite of stated intent

Lines affected:

| Line | Code | Issue |
|------|------|-------|
| 442 | `cost_values = [0.05,0.95,0.48,0]` | Comment says high=preferred, optimizer says high=penalty |
| 826 | `np.full(..., 128, ...)` | Default 128 → moderately penalized |
| 833 | `pred * 255 * cost_values[i]` | Preferred terrain gets highest pixel values = highest penalties |
| 910 | `np.max(costs)` | MAX selects most-preferred pixel, becomes largest penalty |
| 613 | `expected_behav += (max_behav_cost / 255)` | Added to minimization objective |
| 905 | `return marked_img, 0.0` | Out-of-frame = optimal behavioral score |

> **Note:** This may be intentional (the system works in practice), but the semantics are confusing and the comment is misleading. If the optimizer truly minimizes, then low cost_values should correspond to preferred terrain.

## 6.3 Five Concrete Conflict Scenarios

### Scenario 1: "Sidewalk ends" — pavement detection drops to 0

1. CLIPSeg processes image → "Pavement" activation never exceeds 0.1 threshold
2. Pavement mask is all `False` → nothing written for pavement
3. Cost map stays at 128 (default) or gets vegetation/grass values (~10-122)
4. All trajectories have similar behavioral costs → **optimizer becomes indifferent to behavioral guidance**
5. Robot proceeds toward goal ignoring "stay on pavement" entirely. No notification. No slowing down.

### Scenario 2: Surrounded by "avoid" zones

1. "vegetation" (0.05) and "grass" (0.48) activate across entire image
2. Sequential overwriting: vegetation writes ~10, grass overwrites to ~97-122
3. All trajectories have low-to-medium costs
4. No detection that "no preferred terrain exists"
5. Robot navigates through vegetation (least-avoided) with **zero awareness that preferred terrain is absent**

### Scenario 3: Obstacle blocks preferred path (collision avoidance disabled)

1. LiDAR subscription commented out (`behav-planner.py:336`)
2. `expected_collision = 0.0` always (`behav-planner.py:582-583`)
3. `total_cost = 0.0 + expected_progress + 0.0 + expected_behav`
4. Optimizer selects trajectory with best progress regardless of obstacles
5. **Robot drives straight through the obstacle**

### Scenario 4: Forward area entirely high-cost

1. Camera sees only "vegetation" → all forward pixels get cost ~10
2. Out-of-frame trajectories get cost 0.0 (escape hatch)
3. Progress cost dominates (2x distance + 1x heading), so forward trajectories still win
4. But if behavioral cost were higher (Stop gesture), **optimizer could favor backward trajectories**

### Scenario 5: Overlapping rules — "stay on pavement" AND "avoid crowds"

1. Loop processes prompts in index order: vegetation(0), Pavement(1), grass(2), Stop(3)
2. If "Pavement" (cost=0.95) writes first → pixels get ~193
3. If "crowds" (cost=0.05) writes second → **overwrites pavement's 193 with ~10**
4. Pavement preference completely destroyed at crowd locations
5. **Resolution entirely dependent on prompt ordering — no principled conflict resolution**

## 6.4 Dialogue Injection Points

### Point 1: After cost map computation, before optimization
**Location:** `behav-planner.py:837` (after `self.behav_costmap = combined_cost_map`)

**Data available:** Full cost map, CLIPSeg activations, prompt list, cost values

**Trigger condition:** `preferred_ratio = np.sum(costmap > 200) / costmap.size < 0.01`

**Robot asks:** "I see mostly vegetation ahead with no pavement visible. Should I continue through vegetation or wait for you to redirect me?"

**Response modifies:** "continue" → set all costs to 0.5 (neutral). "wait" → set velocity to zero.

### Point 2: After optimization if solution quality is poor
**Location:** `behav-planner.py:543-556` (after local optimization returns)

**Data available:** Optimized parameters, final cost value, trajectory count

**Trigger condition:** `final_cost > QUALITY_THRESHOLD`

**Robot asks:** "All forward paths cross vegetation. The best trajectory has high behavioral cost (0.82). Should I proceed anyway or do you want to give me a new waypoint?"

**Response modifies:** "proceed" → accept trajectory. "new waypoint" → re-enter goal parameters. "stop" → halt.

### Point 3: When CLIPSeg confidence is low
**Location:** `behav-planner.py:822-826` (after sigmoid, before cost map)

**Data available:** `preds` tensor (sigmoid activations for all prompts), per-prompt max/coverage

**Trigger condition:** `torch.max(preds[i]) < 0.3` for all prompts

**Robot asks:** "I'm having trouble identifying terrain types. Confidence for pavement detection is only 12%. Can you confirm: pavement, grass, dirt, or other?"

**Response modifies:** Override CLIPSeg with uniform cost for human-specified terrain type.

### Point 4: When instruction decomposition is ambiguous
**Location:** `instruction-to-behavioral-costs.py:262-266` (after extraction, before cost calc)

**Data available:** `behavioral_action_list`, `behavioral_target_list`, `similarity_scores`

**Trigger condition:** Two targets with opposing costs detected for overlapping spatial regions

**Robot asks:** "Your instruction says 'stay on the path' and 'avoid crowds,' but crowds are on the path. Which takes priority?"

**Response modifies:** Reorder prompts so priority rule is processed last (wins overwriting) or adjust cost values.

### Point 5: When landmark detection fails
**Location:** `landmark_detector_ROS2.py:291-332` (after API response parsing)

**Data available:** `response_json`, `landmark_status` (YES/NO), masked image

**Trigger condition:** `landmark_status == "NO"` or API failure

**Robot asks:** "I cannot find the landmark 'red building' in my camera view. Should I: (1) continue and re-scan in 10s, (2) rotate to scan area, or (3) skip to next instruction?"

**Response modifies:** (1) delay re-trigger, (2) set angular velocity, (3) advance instruction sequence.

## 6.5 Missing Conflict Detection Mechanisms

| What's missing | Where it should be | Impact |
|---|---|---|
| **"All trajectories bad" check** | After optimization (line 543-556) | Robot always executes, never stops to reconsider |
| **Contradictory prompts check** | At init (lines 440-442) or runtime | Overlapping CLIPSeg activations silently fight |
| **Unsatisfiable constraints check** | In image_callback (lines 831-837) | Preferred terrain disappears → planner ignores constraints |
| **Graceful degradation** | In main_loop (lines 482-512) | No speed reduction when uncertain, no progressive relaxation |
| **Human notification system** | Throughout | No `/planner_status` topic, no `/behavioral_conflict` topic, no `/human_query` service |
| **Cost map staleness detection** | In get_traj_behav_cost | `self.cost_map_mutex` declared (line 420) but never acquired |
| **Module integration** | Between instruction-decomposition and planner | Decomposition output never reaches planner — completely disconnected |
| **Mutual exclusion on cost map** | image_callback write / get_traj_behav_cost read | `cost_map_mutex` declared but unused — potential torn reads |
