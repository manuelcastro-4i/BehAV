# Paper Research: Interactive Behavioral Navigation

## 1. BehAV Codebase Analysis — Current State

### 1.1 Instruction Decomposition (`instruction-decomposition/GPT4o/instruction-to-behavioral-costs.py`)

**Pipeline:**
- Single GPT-4 zero-shot call decomposes instructions into 4 fixed categories: landmarks, navigation_actions, behavioral_actions, behavioral_targets
- Cost assignment via similarity matrix (GPT generates NxM matrix) against 4 reference actions:
  - `Stay on = 0, Avoid = 0.5, Yield = 0.7, Stop = 1`
- Argmax selects most similar reference action, assigns its fixed cost
- Parsing via `ast.literal_eval()` with regex fallback — fragile

**Limitations:**
- No temporal sequencing ("first X, then Y")
- No conditional logic ("if X, do Y; else Z")
- No disambiguation or ambiguity detection
- No feedback loop — one-shot decomposition with no validation
- No confidence scores
- Fixed 4 reference categories — cannot express gradations
- GPT output format not guaranteed — parsing failures possible

### 1.2 Landmark Tracking (`landmark-tracking/landmark_detector_ROS2.py`, `landmarkdetector_fastsam.py`)

**Pipeline:**
```
Image → FastSAM (conf=0.4, iou=0.9, imgsz=1024) → All masks
→ Mask merging (distance_threshold=500px) → GPT-4V Vision analysis
→ Binary YES/NO + mask number + distance estimate (string)
```

**Distance estimation:** Entirely delegated to GPT-4V — implicit size-based scaling from reference image at 80m. No explicit algorithm, no camera intrinsics used, no uncertainty bounds.

**Limitations:**
- No confidence scoring (binary YES/NO)
- No temporal tracking (each frame independent)
- Fragile string parsing of GPT responses
- Fixed hyperparameters not adapted to scene
- No occlusion handling
- Distance accuracy estimated at +/-20-30%
- Slow: ~3-5s per frame (API call)
- Model uses `gpt-4-vision-preview` (outdated)

### 1.3 Planning (`planning/behav-planner.py`)

**CLIPSeg Cost Maps:**
```python
prompts = ["vegetation", "Pavement", "grass", "Stop gesture"]
cost_values = [0.05, 0.95, 0.48, 0]  # 0=avoid, 1=prefer, 0.5=neutral
```
- Batch CLIPSeg inference at ~10Hz
- Hard threshold at 0.1 activation
- Last activated prompt overwrites earlier ones (order-dependent!)
- Initial cost map value = 128 (neutral)

**MPC Optimization:**
- Two-stage: Global (GN_ESCH, 300 evals) → Local (LN_BOBYQA, 75 evals)
- 4 decision variables: r, delta, theta, vMax
- Time horizon: 4s with 0.5s steps
- **Behavioral cost = MAX along trajectory** — one bad pixel kills trajectory
- `expected_collision = 0.0` always — NO obstacle avoidance implemented
- If trajectory projects out of camera frame → cost = 0 (can escape constraints)
- No dynamic reconfiguration of parameters

**Critical gaps:**
- No collision avoidance (LiDAR subscription commented out)
- No fallback when constraints are unsatisfiable
- No conflict detection between behavioral rules
- Pipeline is strictly feed-forward — no feedback from execution

---

## 2. State of the Art (2024-2026)

### 2.1 Human-Robot Dialogue in Navigation

**KnowNo** (Ren et al., CoRL 2023)
- Conformal prediction to quantify LLM planner uncertainty
- Triggers human help when prediction set has >1 option
- Statistical guarantees on task completion
- Limitation: frequent superfluous queries
- https://arxiv.org/abs/2307.01928 | https://robot-help.github.io/

**Introspective Planning** (Liang et al., NeurIPS 2024)
- Extends KnowNo with introspection step aligning uncertainty with task ambiguity
- Knowledge base of introspective reasoning examples
- Drastically reduces overasking while maintaining guarantees
- https://arxiv.org/abs/2402.02205 | https://github.com/kevinliang888/IntroPlan

**DRAGON** (Liu et al., RA-L 2024)
- Bidirectional voice dialogue for visually impaired navigation
- Novel landmark mapping with free-form language commands
- Validated in user study with blindfolded participants
- https://arxiv.org/abs/2307.06924 | https://github.com/Shuijing725/Dragon_Wayfinding

**ORION / Think, Act, and Ask** (Dai et al., ICRA 2024)
- Zero-shot Interactive Personalized Object Navigation (ZIPON)
- LLM decides: perceive, navigate, or communicate
- Interactive agents significantly outperform non-interactive
- https://arxiv.org/abs/2310.07968 | https://github.com/sled-group/navchat

**DriVLMe** (Huang et al., IROS 2024)
- Video-language-model agent for human-vehicle dialogue
- Long-horizon navigation with free-form dialogue
- Handles unexpected situations mid-execution
- https://arxiv.org/abs/2406.03008

**DOROTHIE** (Ma et al., Findings of EMNLP 2022)
- Interactive simulation for spoken dialogue with autonomous driving agents
- 183 trials, 8415 utterances, 18.7 hours of control streams
- https://arxiv.org/abs/2210.12511

**HSAC-LLM** (Wen et al., IROS 2025)
- Deep RL + LLM for bidirectional natural language with pedestrians
- Proactive robot communication for collision avoidance
- Validated in 2D sim, Gazebo, and real-world
- https://arxiv.org/abs/2409.04965 | https://hsacllm.github.io/

### 2.2 VLM/LLM-based Robot Navigation

**NaVid** (Zhang et al., RSS 2024)
- Video-based VLM navigation with only monocular RGB — no maps/odometry
- 510k navigation + 763k web data samples
- Strong sim-to-real transfer
- https://pku-epic.github.io/NaVid/

**Uni-NaVid** (Zhang et al., RSS 2025)
- Unified video-based VLA: VLN + ObjectNav + EQA + human-following
- Online token merge for long video streams at 5 Hz
- 3.6M navigation samples
- https://arxiv.org/abs/2412.06224

**NaVILA** (Cheng et al., RSS 2025)
- VLA + locomotion skills for legged robots (Go2 and H1!)
- Language-as-action: "moving forward 75cm"
- 88% success rate real-world, 17% gain on VLN benchmarks
- https://arxiv.org/abs/2412.04453 | https://github.com/AnjieCheng/NaVILA

**ViLA** (Hu et al., 2024)
- GPT-4V as external planner with chain-of-thought from visual observations
- Closed-loop: execute → observe → replan
- Flexible multimodal goal specification
- https://arxiv.org/abs/2311.17842 | https://robot-vila.github.io/

**VL-Nav** (Du et al., 2025)
- Real-time zero-shot VLN at 30 Hz on Jetson Orin NX
- Curiosity-driven exploration with GMM-based spatial scoring
- 86.3% success rate, outperforms prior by 44.15%
- https://arxiv.org/abs/2502.00931 | https://sairlab.org/vlnav/

**ViNT** (Shah et al., CoRL 2023 / ICRA 2024 finalist)
- Foundation model for visual navigation, embodiment-agnostic
- 40% zero-shot transfer, 80% with <1h fine-tuning
- https://arxiv.org/abs/2306.14846

**NoMaD** (Shah et al., ICRA 2024 finalist)
- Goal-conditioned action diffusion model for navigation
- Unifies exploration and goal-directed behavior
- https://arxiv.org/abs/2310.07896

**VLM-GroNav** (Elnoor et al., 2024)
- VLM + proprioception for outdoor navigation on legged/wheeled robots
- Assesses terrain traversability with proprioceptive feedback
- Global planner (large VLM on aerial images) + local planner (compact VLM + proprioception)
- 50% improvement in navigation success
- https://arxiv.org/abs/2409.20445

### 2.3 Scene Graph-Based Navigation

**SG-Nav** (Yin et al., NeurIPS 2024)
- Online 3D scene graphs + hierarchical chain-of-thought LLM
- Re-perception mechanism to correct detection errors
- First zero-shot method matching supervised on MP3D
- https://arxiv.org/abs/2410.08189

**HOV-SG** (Werby et al., RSS 2024)
- Hierarchical open-vocabulary 3D scene graphs (floor/room/object)
- 75% reduction in representation size
- https://arxiv.org/abs/2403.17846

**ConceptGraphs** (Gu et al., ICRA 2024)
- Open-vocabulary graph-structured 3D scene representations
- CLIP embeddings + text captions per object
- https://arxiv.org/abs/2309.16650 | https://concept-graphs.github.io/

**VLFM** (Yokoyama et al., ICRA 2024)
- Vision-language frontier maps for object goal navigation
- Deployed on Boston Dynamics Spot
- https://arxiv.org/abs/2312.03275

### 2.4 Behavioral / Social Navigation

**VLM-Social-Nav** (Raj et al., RA-L 2024)
- YOLO detection + VLM socially compliant behavior scores
- VLM cost term integrated into planner
- 27.38% success improvement, 19.05% collision reduction
- No training data required
- https://arxiv.org/abs/2404.00210

**"Language as Cost"** (2025)
- VLMs proactively identify and map hazards from language descriptions
- Cost maps from language-specified hazard descriptions
- https://arxiv.org/html/2508.03138

**Social Navigation Survey** (Singamaneni et al., IJRR 2024)
- Comprehensive taxonomy: proxemics, formations, social forces, learning
- https://journals.sagepub.com/doi/10.1177/02783649241230562

**Social Navigation Review** (Frontiers 2025)
- OLiVia-Nav distills VLM social context into lightweight encoders
- GSON uses VLMs to detect social groups for MPC
- https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1658643/full

### 2.5 Instruction Decomposition Improvements

**GRID** (Zeng et al., IROS 2024)
- LLMs + Graph Attention Networks on scene graphs
- Outperforms GPT-4 by 43.6% in task accuracy
- Lightweight, locally deployable
- https://arxiv.org/abs/2309.07726

**BrainBody-LLM** (Bhat et al., 2025)
- Two-LLM hierarchy: Brain (high-level reasoning) + Body (low-level commands)
- Closed-loop feedback relays execution errors for replanning
- 17% improvement over baselines
- https://arxiv.org/abs/2402.08546

**SayPlan** (Rana et al., CoRL 2023)
- LLM grounded by 3D scene graphs for large-scale task planning
- Iterative replanning with simulator feedback
- https://arxiv.org/abs/2307.06135

**EmbodiedRAG** (Booker et al., 2024)
- RAG on 3D scene graphs — 10x token reduction, 70% faster planning
- Deployed on quadruped with manipulator
- https://arxiv.org/abs/2410.23968

**LLM-Grounder** (Yang et al., ICRA 2024)
- LLM decomposes complex queries → grounds in 3D (OpenScene/LERF)
- State-of-the-art zero-shot on ScanRefer
- https://arxiv.org/abs/2309.12311

### 2.6 Landmark Detection / Segmentation Evolution

**Grounding DINO** (Liu et al., ECCV 2024)
- Open-set language-conditioned detector
- https://github.com/IDEA-Research/GroundingDINO

**Grounding DINO 1.5** (Ren et al., 2024)
- Pro: 54.3 AP COCO, 55.7 zero-shot LVIS
- Edge: 75.2 FPS with TensorRT
- https://arxiv.org/abs/2405.10300

**SAM 2** (Meta, July 2024)
- Unified image + video segmentation, 6x faster than SAM, ~44 FPS
- Temporal tracking across frames
- https://arxiv.org/html/2408.00714v1

**Grounded SAM 2** (IDEA Research, 2024)
- Grounding DINO + SAM 2 integrated
- Open-vocabulary detection + segmentation + video tracking
- https://github.com/IDEA-Research/Grounded-SAM-2

**MobileSAM** (Zhang et al., 2023)
- 60x smaller than SAM, ~10ms per image
- 5x faster than FastSAM, 7x smaller
- https://arxiv.org/abs/2306.14289

**Perception comparison for BehAV:**

| Model | Speed | Quality | Video Tracking | Open-Vocab | On-Device |
|-------|-------|---------|---------------|------------|-----------|
| GPT-4V + SAM (current) | Very slow (API) | High | No | Yes (via GPT) | No |
| Grounding DINO 1.5 Edge + SAM 2 | 75+44 FPS | High | Yes | Yes | Yes |
| Grounded SAM 2 | Real-time | High | Yes | Yes | Partial |
| OWLv2 + MobileSAM | Real-time | Med-High | No | Yes | Yes |
| FastSAM (current) | ~100 img/s | Medium | No | Prompt-guided | Yes |

---

## 3. Research Gap Analysis

### What exists:
- "Ask for help" with conformal prediction (KnowNo, Introspective Planning) — indoor manipulation/navigation
- Bidirectional dialogue for navigation (DRAGON, ORION, DriVLMe) — assistive/driving
- VLM-based social navigation cost maps (VLM-Social-Nav) — indoor social norms
- End-to-end VLM navigation (NaVid, NaVILA) — no behavioral rules

### What does NOT exist (our gap):
- **Dialogue for behavioral rule conflicts in outdoor navigation**
- **Conformal prediction applied to behavioral cost map ambiguity**
- **Robot communicating about terrain/behavioral constraint violations**
- **Interactive relaxation of behavioral rules with human oversight**
- **Outdoor legged robot that explains its behavioral navigation decisions**

---

## 4. Paper Proposal

### Title options:
- "Interactive Behavioral Navigation: Enabling Quadruped Robots to Communicate When Behavioral Rules Conflict"
- "BehAV-Talk: Human-Robot Dialogue for Behavioral Navigation Under Uncertainty"
- "When Rules Break: Interactive Behavioral Navigation with Human-in-the-Loop Conflict Resolution"

### Proposed contributions:
1. **Behavioral dialogue framework**: When/how a robot should communicate during behavioral navigation (3 phases: obstacles, paths, ambiguity)
2. **Behavioral conflict detection**: Method to detect unsatisfiable/ambiguous behavioral constraints using LLM introspection
3. **Perception upgrades**: Grounding DINO + SAM2 for real-time landmark tracking + structured instruction decomposition
4. **Evaluation**: Scenarios in Go2 Gazebo simulation with metrics: success rate, human queries, mission time, rule violations

### Differentiator vs. state of the art:
KnowNo/Introspective Planning handle "which object to grasp" ambiguity. We handle "which behavioral rule to violate when there's no valid path" — fundamentally different domain.

The modular pipeline (vs. end-to-end) is a feature, not a bug: interpretability enables communication. An end-to-end model can't explain WHY it chose a path.

### Risk:
Field moving toward end-to-end (NaVILA, NaVid). Counter: emphasize interpretability + communicability as the research contribution, not the pipeline architecture itself.

### Three dialogue phases:

**(a) Obstacle identification:**
- Robot detects obstacle not in behavioral cost map
- Queries human: "There's a puddle blocking the sidewalk. Go around left (through grass) or right (through parking)?"
- Grounded in conformal prediction: ask only when uncertain

**(b) Path negotiation:**
- Robot proposes trajectory, human can redirect
- "I'm planning to go straight along the path. There's a shorter route through the park — should I take it?"
- Requires basic spatial reasoning about alternatives

**(c) Behavioral rule conflict:**
- Core novelty: detect when rules are contradictory or unsatisfiable
- "The instruction says 'stay on pavement' but the pavement ends here. Can I cross the grass to reach the next sidewalk?"
- Framework: quantify violation severity → consult human only for significant violations
- Inspired by Introspective Planning but applied to behavioral cost maps

### Technical improvements needed:

**Instruction decomposition:**
1. Structured JSON output (schema-constrained) instead of ast.literal_eval
2. Hierarchical temporal decomposition: "first X, then Y, finally Z"
3. Confidence scores per extracted component
4. Ambiguity detection: flag unclear references before committing

**Landmark tracking:**
1. Replace GPT-4V + FastSAM with Grounding DINO 1.5 + SAM 2
2. Temporal tracking (SAM 2 video mode) instead of per-frame detection
3. Explicit distance estimation with uncertainty bounds
4. Confidence scoring for detections

**Planning:**
1. Integrate LiDAR for obstacle avoidance (currently disabled)
2. Soft behavioral costs (weighted average, not max)
3. Constraint relaxation with human consultation
4. Instruction-aware CLIPSeg prompts (from decomposition, not hardcoded)

---

## 5. Key References (Curated)

### Must-cite (directly relevant):
- BehAV original paper: https://arxiv.org/pdf/2409.16484
- KnowNo (Ren et al., CoRL 2023): https://arxiv.org/abs/2307.01928
- Introspective Planning (Liang et al., NeurIPS 2024): https://arxiv.org/abs/2402.02205
- VLM-Social-Nav (Raj et al., RA-L 2024): https://arxiv.org/abs/2404.00210
- ORION (Dai et al., ICRA 2024): https://arxiv.org/abs/2310.07968
- NaVILA (Cheng et al., RSS 2025): https://arxiv.org/abs/2412.04453

### Should-cite (context):
- DRAGON (Liu et al., RA-L 2024): https://arxiv.org/abs/2307.06924
- HSAC-LLM (Wen et al., IROS 2025): https://arxiv.org/abs/2409.04965
- NaVid (Zhang et al., RSS 2024): https://pku-epic.github.io/NaVid/
- VLM-GroNav (Elnoor et al., 2024): https://arxiv.org/abs/2409.20445
- ConceptGraphs (Gu et al., ICRA 2024): https://arxiv.org/abs/2309.16650
- SG-Nav (Yin et al., NeurIPS 2024): https://arxiv.org/abs/2410.08189
- Grounding DINO 1.5 (Ren et al., 2024): https://arxiv.org/abs/2405.10300
- SAM 2 (Meta, 2024): https://arxiv.org/html/2408.00714v1
- GRID (Zeng et al., IROS 2024): https://arxiv.org/abs/2309.07726
- BrainBody-LLM (Bhat et al., 2025): https://arxiv.org/abs/2402.08546
- Language as Cost (2025): https://arxiv.org/html/2508.03138
- Social Navigation Survey (IJRR 2024): https://journals.sagepub.com/doi/10.1177/02783649241230562

### Nice-to-cite (broader context):
- DriVLMe (IROS 2024): https://arxiv.org/abs/2406.03008
- DOROTHIE (EMNLP 2022): https://arxiv.org/abs/2210.12511
- ViLA (2024): https://arxiv.org/abs/2311.17842
- VL-Nav (2025): https://arxiv.org/abs/2502.00931
- ViNT (CoRL 2023): https://arxiv.org/abs/2306.14846
- NoMaD (ICRA 2024): https://arxiv.org/abs/2310.07896
- HOV-SG (RSS 2024): https://arxiv.org/abs/2403.17846
- VLFM (ICRA 2024): https://arxiv.org/abs/2312.03275
- SayPlan (CoRL 2023): https://arxiv.org/abs/2307.06135
- EmbodiedRAG (2024): https://arxiv.org/abs/2410.23968
- LLM-Grounder (ICRA 2024): https://arxiv.org/abs/2309.12311
- Grounded SAM 2: https://github.com/IDEA-Research/Grounded-SAM-2
- MobileSAM: https://arxiv.org/abs/2306.14289
- Uni-NaVid (RSS 2025): https://arxiv.org/abs/2412.06224

---

## 6. Phase (c) Deep Dive — Behavioral Conflict Analysis in BehAV Code

This section maps exactly WHERE and HOW behavioral conflicts manifest in the BehAV codebase, with line numbers, execution traces, and dialogue injection points.

### 6.1 Cost Map Conflict Mechanics

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

### 6.2 Inverted Cost Semantics Bug

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

### 6.3 Five Concrete Conflict Scenarios

#### Scenario 1: "Sidewalk ends" — pavement detection drops to 0

1. CLIPSeg processes image → "Pavement" activation never exceeds 0.1 threshold
2. Pavement mask is all `False` → nothing written for pavement
3. Cost map stays at 128 (default) or gets vegetation/grass values (~10-122)
4. All trajectories have similar behavioral costs → **optimizer becomes indifferent to behavioral guidance**
5. Robot proceeds toward goal ignoring "stay on pavement" entirely. No notification. No slowing down.

#### Scenario 2: Surrounded by "avoid" zones

1. "vegetation" (0.05) and "grass" (0.48) activate across entire image
2. Sequential overwriting: vegetation writes ~10, grass overwrites to ~97-122
3. All trajectories have low-to-medium costs
4. No detection that "no preferred terrain exists"
5. Robot navigates through vegetation (least-avoided) with **zero awareness that preferred terrain is absent**

#### Scenario 3: Obstacle blocks preferred path (collision avoidance disabled)

1. LiDAR subscription commented out (`behav-planner.py:336`)
2. `expected_collision = 0.0` always (`behav-planner.py:582-583`)
3. `total_cost = 0.0 + expected_progress + 0.0 + expected_behav`
4. Optimizer selects trajectory with best progress regardless of obstacles
5. **Robot drives straight through the obstacle**

#### Scenario 4: Forward area entirely high-cost

1. Camera sees only "vegetation" → all forward pixels get cost ~10
2. Out-of-frame trajectories get cost 0.0 (escape hatch)
3. Progress cost dominates (2x distance + 1x heading), so forward trajectories still win
4. But if behavioral cost were higher (Stop gesture), **optimizer could favor backward trajectories**

#### Scenario 5: Overlapping rules — "stay on pavement" AND "avoid crowds"

1. Loop processes prompts in index order: vegetation(0), Pavement(1), grass(2), Stop(3)
2. If "Pavement" (cost=0.95) writes first → pixels get ~193
3. If "crowds" (cost=0.05) writes second → **overwrites pavement's 193 with ~10**
4. Pavement preference completely destroyed at crowd locations
5. **Resolution entirely dependent on prompt ordering — no principled conflict resolution**

### 6.4 Dialogue Injection Points

#### Point 1: After cost map computation, before optimization
**Location:** `behav-planner.py:837` (after `self.behav_costmap = combined_cost_map`)

**Data available:** Full cost map, CLIPSeg activations, prompt list, cost values

**Trigger condition:** `preferred_ratio = np.sum(costmap > 200) / costmap.size < 0.01`

**Robot asks:** "I see mostly vegetation ahead with no pavement visible. Should I continue through vegetation or wait for you to redirect me?"

**Response modifies:** "continue" → set all costs to 0.5 (neutral). "wait" → set velocity to zero.

#### Point 2: After optimization if solution quality is poor
**Location:** `behav-planner.py:543-556` (after local optimization returns)

**Data available:** Optimized parameters, final cost value, trajectory count

**Trigger condition:** `final_cost > QUALITY_THRESHOLD`

**Robot asks:** "All forward paths cross vegetation. The best trajectory has high behavioral cost (0.82). Should I proceed anyway or do you want to give me a new waypoint?"

**Response modifies:** "proceed" → accept trajectory. "new waypoint" → re-enter goal parameters. "stop" → halt.

#### Point 3: When CLIPSeg confidence is low
**Location:** `behav-planner.py:822-826` (after sigmoid, before cost map)

**Data available:** `preds` tensor (sigmoid activations for all prompts), per-prompt max/coverage

**Trigger condition:** `torch.max(preds[i]) < 0.3` for all prompts

**Robot asks:** "I'm having trouble identifying terrain types. Confidence for pavement detection is only 12%. Can you confirm: pavement, grass, dirt, or other?"

**Response modifies:** Override CLIPSeg with uniform cost for human-specified terrain type.

#### Point 4: When instruction decomposition is ambiguous
**Location:** `instruction-to-behavioral-costs.py:262-266` (after extraction, before cost calc)

**Data available:** `behavioral_action_list`, `behavioral_target_list`, `similarity_scores`

**Trigger condition:** Two targets with opposing costs detected for overlapping spatial regions

**Robot asks:** "Your instruction says 'stay on the path' and 'avoid crowds,' but crowds are on the path. Which takes priority?"

**Response modifies:** Reorder prompts so priority rule is processed last (wins overwriting) or adjust cost values.

#### Point 5: When landmark detection fails
**Location:** `landmark_detector_ROS2.py:291-332` (after API response parsing)

**Data available:** `response_json`, `landmark_status` (YES/NO), masked image

**Trigger condition:** `landmark_status == "NO"` or API failure

**Robot asks:** "I cannot find the landmark 'red building' in my camera view. Should I: (1) continue and re-scan in 10s, (2) rotate to scan area, or (3) skip to next instruction?"

**Response modifies:** (1) delay re-trigger, (2) set angular velocity, (3) advance instruction sequence.

### 6.5 Missing Conflict Detection Mechanisms

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

---

## 7. Phase (c) Deep Dive — Expanded Literature Review (55+ Papers)

### 7.1 Constraint Conflict/Violation Detection in Robot Planning

**Lahijanian, Almagor, Fried, Kavraki, Vardi (2015).** "This Time the Robot Settles for a Cost: A Quantitative Approach to Temporal Logic Planning with Partial Satisfaction." AAAI 2015. Quantifies partial satisfaction of co-safe LTL when full specification is infeasible. Assigns costs to proposition violations and synthesizes trajectories with optimal satisfaction. Directly addresses what to do when behavioral constraints become infeasible.
https://ojs.aaai.org/index.php/AAAI/article/view/9670

**Raman, Kress-Gazit (2013).** "Explaining Impossible High-Level Robot Behaviors." IEEE T-RO 29(1). Algorithm for identifying unsatisfiable/unrealizable components in temporal logic specs + interactive game for exploring failure reasons. Foundational work on detecting when robot specifications conflict with environmental realities.
https://ieeexplore.ieee.org/document/6301745

**Raman, Lignos, Finucane, Lee, Marcus, Kress-Gazit (2013).** "Sorry Dave, I'm Afraid I Can't Do That: Explaining Unachievable Robot Tasks Using Natural Language." RSS 2013. Integrates formal methods with NL to provide human-understandable feedback when specifications cannot be achieved. Directly relevant to explaining behavioral constraint violations.
https://www.roboticsproceedings.org/rss09/p23.html

**Ren, Ren, Muvvala, Luo, Lahijanian (2024).** "LTL-D*: Incrementally Optimal Replanning for Feasible and Infeasible Tasks in LTL Specifications." IROS 2024. Incremental replanning for both feasible and infeasible LTL specs in dynamic environments, using a distance metric for violation degree. Directly addresses replanning when behavioral constraints are violated.
https://arxiv.org/abs/2404.01219

**Kerrigan, Maciejowski (2000).** "Soft Constraints and Exact Penalty Functions in Model Predictive Control." UKACC 2000. Foundational work on recovering MPC feasibility when constraints are violated, using exact penalty functions and slack variables. BehAV uses MPC, making this directly relevant.

**Yang, Raman, Shah, Tellex (2024).** "Plug in the Safety Chip: Enforcing Constraints for LLM-driven Robot Agents." ICRA 2024. Framework with verifiable guarantees that safety constraints won't be violated by LLM robots, with reprompting when constraints would be violated.
https://arxiv.org/abs/2309.09919

**Wu, Xiong, Hu, et al. (2025).** "SELP: Generating Safe and Efficient Task Plans for Robot Agents with LLMs." ICRA 2025. Combines equivalence voting, constrained decoding via Buchi automata, and fine-tuning to ensure LLM plans conform to LTL safety specifications. 10.8% safety improvement.
https://arxiv.org/abs/2409.19471

### 7.2 LLM/VLM Introspection and Uncertainty Quantification

**Ren et al. (2023).** "Robots That Ask For Help: Uncertainty Alignment for LLM Planners (KnowNo)." CoRL 2023 (Best Student Paper). Conformal prediction to align LLM uncertainty; robots know when they don't know and ask for help. Statistical guarantees on task completion while minimizing help requests. **Most directly relevant paper.**
https://arxiv.org/abs/2307.01928

**Liang et al. (2024).** "Introspective Planning: Aligning Robots' Uncertainty with Inherent Task Ambiguity." NeurIPS 2024. Retrieval-augmented introspective reasoning + conformal prediction for tighter confidence bounds. Reduces unnecessary queries while maintaining guarantees.
https://arxiv.org/abs/2402.06529

**Huang, Xia, et al. (2023).** "Inner Monologue: Embodied Reasoning through Planning with Language Models." CoRL 2023. Environment feedback (success detection, scene description, human interaction) as inner monologue enabling LLMs to plan in robotic scenarios. Relevant to introspecting on constraint violations.
https://arxiv.org/abs/2207.05608

**Ahn, Brohan, Brown, et al. (2022).** "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)." CoRL 2023. LLM proposes actions, affordance functions determine feasibility. Relevant to detecting when behavioral actions are infeasible.
https://arxiv.org/abs/2204.01691

**Liu, Bahety, Song (2023).** "REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction." CoRL 2023. VLMs with hierarchical summaries for failure reasoning + corrective replanning. Relevant to detecting/explaining behavioral constraint violations.
https://arxiv.org/abs/2306.15724

### 7.3 Norm Compliance and Violation in Autonomous Agents

**Neufeld (2023).** "Norm Compliance for Reinforcement Learning Agents." PhD Thesis, TU Wien. Introduces the Normative Supervisor: defeasible deontic logic integrated with RL for real-time normative compliance. Handles contrary-to-duty norms (what to do AFTER a violation), directly modeling behavioral rule conflicts.
https://repositum.tuwien.at/handle/20.500.12708/177391

**Neufeld, Gangl, Murano, Governatori (2022).** "On Normative Reinforcement Learning via Safe Reinforcement Learning." PRIMA 2022. Demonstrates standard safe RL with LTL is insufficient for normative reasoning; proposes defeasible deontic logic for handling norm conflicts, violations, and contrary-to-duty obligations.
https://link.springer.com/chapter/10.1007/978-3-031-21203-1_5

**Neufeld, Gangl, Murano, Governatori (2024).** "Learning Normative Behaviour Through Automated Theorem Proving." KI - Kunstliche Intelligenz. Violation counting for normative conflicts + contrary-to-duty norms, broadening the range of normative systems that can be learned.
https://link.springer.com/article/10.1007/s13218-024-00844-x

**Noothigattu et al. (2022).** "Enforcing Ethical Goals over Reinforcement-Learning Policies." Ethics and Information Technology. Framework for enforcing ethical constraints over RL policies when ethical goals conflict with task objectives.
https://link.springer.com/article/10.1007/s10676-022-09665-8

**Dagstuhl Seminar (2023).** "Normative Reasoning for AI (Dagstuhl Seminar 23151)." Dagstuhl Reports 13(4). Comprehensive seminar on open problems in combining formal normative reasoning with ML for autonomous systems.
https://drops.dagstuhl.de/entities/document/10.4230/DagRep.13.4.1

### 7.4 Human-in-the-Loop Constraint Modification

**Cui, Karamcheti, Sadigh, Losey (2023).** "No, to the Right: Online Language Corrections for Robotic Manipulation via Shared Autonomy (LILAC)." HRI 2023. Real-time NL corrections during execution refine the human's control space. Directly relevant to humans modifying behavioral constraints through dialogue.
https://arxiv.org/abs/2301.02555

**Chiou, Hawes, Stolkin (2021).** "Mixed-Initiative Variable Autonomy for Remotely Operated Mobile Robots (EMICS)." ACM THRI 10(4). Switching between autonomy levels initiated by either human or robot. Mixed-initiative outperforms single-mode. Relevant to switching between autonomous rule-following and human-guided constraint relaxation.
https://dl.acm.org/doi/10.1145/3472206

**Yu, Albert, et al. (2025).** "Mixed-Initiative Dialog for Human-Robot Collaborative Manipulation (MICoBot)." RSS 2025 Workshop. LLM-based meta-planning with three-level decision architecture. Both human and robot take initiative in proposing/accepting/rejecting task allocation.
https://arxiv.org/abs/2508.05535

**Li, Snavely, et al. (2024).** "Mixed-Initiative Human-Robot Teaming under Suboptimality with Online Bayesian Adaptation." 2024. Online Bayesian inference adapts to human willingness to comply with robot assistance.
https://arxiv.org/abs/2403.16178

**Ryu, Matsubara (2023).** "Task-oriented Dialogues with Mixed-Initiative Interactions." IJCAI 2023. Mixed-initiative strategy-based prompting for proactive dialogue, with policy planning for conversation turns.
https://www.ijcai.org/proceedings/2023/0583.pdf

### 7.5 Cost Map Adaptation and Dynamic Replanning

**Mao, Chung, Stone, Xiao (2024).** "PACER: Preference-conditioned All-terrain Costmap Generation." IEEE RA-L 2025. Costmaps from BEV images conditioned on user preference context, enabling rapid adaptation to new terrain preferences. **Directly relevant to dynamically modifying BehAV behavioral cost maps from human feedback.**
https://arxiv.org/abs/2410.23488

**Triest et al. (2023).** "Learning Risk-Aware Costmaps via Inverse RL for Off-Road Navigation." ICRA 2023. Max-entropy IRL for deep uncertainty-aware cost functions with parallel MPC. IRL outperforms occupancy-based baselines significantly.
https://arxiv.org/abs/2302.00134

**Goel et al. (2022).** "Predicting Dense and Context-aware Cost Maps for Semantic Robot Navigation." IROS 2022 Workshop. DNN predicts dense cost maps with semantic context for object-goal navigation, used with sampling-based MPC.
https://arxiv.org/abs/2210.08952

**Elnoor et al. (2024).** "VLM-GroNav: Robot Navigation Using Physically Grounded VLMs in Outdoor Environments." IROS 2024. VLMs grounded with proprioceptive data for dynamic terrain traversability updates. Validated on Ghost Vision 60 + Clearpath Husky. 50% improvement.
https://arxiv.org/abs/2409.20445

**Siva et al. (2024).** "Self-reflective Terrain-aware Robot Adaptation for Consistent Off-road Ground Navigation." IJRR 2024. Online self-supervised adaptation + meta-learning for terrain cost estimation using proprioceptive feedback.
https://journals.sagepub.com/doi/10.1177/02783649231225243

**Zhang et al. (2024).** "Learning-based Traversability Costmap for Autonomous Off-road Navigation." 2024. RGB-D + velocity → traversability costmaps with risk-aware labeling from proprioception.
https://arxiv.org/abs/2406.08187

### 7.6 Explainable Robot Navigation

**Halilovic (2025).** "Explainable Robot Navigation." AAAI 2025 Doctoral Consortium. HiXRoN: Hierarchical eXplainable Robot Navigation combining ML with symbolic reasoning for multimodal, personalized explanations. **Directly relevant to explaining behavioral rule conflicts.**
https://ojs.aaai.org/index.php/AAAI/article/view/35208

**Halilovic, Krivic, Canal (2024).** "Towards Probabilistic Planning of Explanations for Robot Navigation." arXiv 2024. RDDL-based framework for planning explanations tailored to user preferences (textual vs. visual, rich vs. poor, local vs. global).
https://arxiv.org/abs/2411.05022

**LeMasurier, Gautam, Han, Crandall, Yanco (2024).** "Reactive or Proactive? How Robots Should Explain Failures." HRI 2024. **Proactive explanations (predicting issues in advance) foster higher trust and perceived intelligence than reactive ones.** Supports our design for proactive conflict communication.
https://dl.acm.org/doi/10.1145/3610977.3634963

**Tagliamonte, Maccaline (2024).** "A Generalizable Architecture for Explaining Robot Failures." UMass Lowell. Template-based NLG for robot failure explanations, adaptable to different platforms and failure types.

**Halilovic, Krivic (2024).** "Towards a Holistic Framework for Explainable Robot Navigation." Springer LNCS. Unified framework integrating explanation modalities, temporal strategies, and personalization.
https://link.springer.com/chapter/10.1007/978-3-031-55000-3_15

### 7.7 Failure Recovery and Help-Seeking in Robotics

**Chen et al. (2024).** "Automating Robot Failure Recovery Using VLMs With Optimized Prompts." IROS 2024. VLMs with optimized prompts for automatic failure detection and recovery. 65.78% accuracy improvement over unoptimized prompts.
https://arxiv.org/abs/2409.03966

**Li et al. (2024).** "Guardian: Detecting Robotic Planning and Execution Errors with VLMs." arXiv 2024. Fine-tuned VLM for failure detection as VQA over task instructions and multi-view images. SOTA across 4 benchmarks.
https://arxiv.org/abs/2512.01946

**Tagliamonte et al. (2025).** "Adapting Robotic Explanations for Robotic Failures in Human Robot Collaboration." HRI 2025. Adapts failure explanations based on observed human behavior via multimodal prediction of user confusion.
https://dl.acm.org/doi/10.5555/3721488.3721795

**Kim et al. (2024).** "The Dilemma of Decision-Making in the Real World: When Robots Struggle Due to Situational Constraints." arXiv 2024. Examines scenarios where robots face decision-making dilemmas from conflicting situational constraints and must interact with humans. **Directly relevant.**
https://arxiv.org/abs/2412.01744

**Ma et al. (2025).** "AINav: LLM-Based Adaptive Interactive Navigation." arXiv 2025. **Quadruped robot** with LLM-based planning that proactively interacts with environments, with adaptive replanning under partial observability. Demonstrates help-seeking in quadruped navigation.
https://arxiv.org/abs/2503.22942

### 7.8 Ethical Decision-Making in Autonomous Systems

**Awad, Dsouza, Kim, et al. (2018).** "The Moral Machine Experiment." Nature 563, 59-64. 40 million ethical decisions from millions of people about AV dilemmas. Reveals global moral preferences and cultural variations. Foundational for ethical trade-offs in autonomous systems.
https://www.nature.com/articles/s41586-018-0637-6

**Chiu, Jiang, et al. (2024).** "DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life." ICLR 2025. 1,360 everyday moral dilemmas; finds LLMs have implicit value priorities that resist steering via system prompts. Relevant to understanding LLM-based planners handling behavioral conflicts.
https://arxiv.org/abs/2410.02683

**Charisi, Liem, Gomez (2024).** "Human-Robot Dialogue that Elicits Alignment of Moral Principles For Driverless Vehicles." HRI 2024 Companion. Dialogue system for eliciting ethical preferences from users to align AV behavior. **Directly relevant to using dialogue to resolve behavioral rule conflicts.**
https://dl.acm.org/doi/10.1145/3610978.3640641

**Geisslinger, Poszler, Betz, Lienkamp (2024).** "Ethical Decision-Making for Self-Driving Vehicles." Science and Engineering Ethics. Model for ethical decision-making with explicit moral attribute definitions, bridging abstract principles and implementable algorithms.
https://link.springer.com/article/10.1007/s11948-024-00513-0

### 7.9 Conformal Prediction in Robotics

**Lindemann, Cleaveland, Shim, Pappas (2023).** "Safe Planning in Dynamic Environments Using Conformal Prediction." IEEE RA-L 8(8). MPC with conformal prediction regions for provably safe planning. Compatible with RNNs/LSTMs. **Directly relevant to adding uncertainty-aware safety to BehAV's MPC.**
https://arxiv.org/abs/2210.10254

**Lindemann, Zhao, Yu, Pappas, Deshmukh (2024).** "Formal Verification and Control with Conformal Prediction." IEEE CSM 2024. **Most complete survey** of CP for autonomous system verification and safe control, from simple navigation to temporal logic missions.
https://arxiv.org/abs/2409.00536

**Sheng, Yu, Parker, Kwiatkowska, Feng (2024).** "Safe POMDP Online Planning via Adaptive Conformal Prediction." IEEE RA-L 9(11). Adaptive CP for safety shields in POMDP planning with hundreds of dynamic agents.
https://arxiv.org/abs/2404.15557

**Dixit, Lindemann, Wei, et al. (2023).** "Adaptive Conformal Prediction for Motion Planning among Dynamic Agents." L4DC 2023. Adaptive CP for online trajectory prediction uncertainty, enabling safe planning with non-exchangeable data.
https://proceedings.mlr.press/v211/dixit23a/dixit23a.pdf

**Kumar et al. (2025).** "Learnable Conformal Prediction with Context-Aware Nonconformity Functions for Robotic Planning." arXiv 2025. Neural nonconformity functions with geometric, semantic, and task-specific features. Addresses CP conservatism.
https://arxiv.org/abs/2509.21955

**Ren et al. (2024).** "Probabilistically Correct Language-based Multi-Robot Planning using Conformal Prediction." arXiv 2024. CP for multi-robot LLM planning; each robot determines when uncertain and seeks help with user-specified success rates.
https://arxiv.org/abs/2402.15368

### 7.10 Mixed-Initiative Interaction for Navigation

**Zhang et al. (2024).** "HSAC-LLM: Socially-Aware Robot Navigation with Bidirectional NL Using LLMs." 2024. Deep RL + LLM for bidirectional NL during navigation. Robot proactively communicates with pedestrians to determine avoidance strategies. **Closest work on robot-initiated dialogue for navigation conflicts.**
https://arxiv.org/abs/2409.04965

**Liu, Yang, et al. (2023).** "Lang2LTL: Grounding Complex NL Commands for Temporal Tasks." CoRL 2023. LLMs ground NL navigation commands to LTL specs in novel environments. Relevant to grounding behavioral rules in formal specs that can be checked for conflicts.
https://arxiv.org/abs/2302.11649

**Shah, Osinski, Ichter, Levine (2023).** "LM-Nav: Robotic Navigation with Large Pre-Trained Models." CoRL 2023. LLM (landmark extraction) + VLM (grounding) + VNM (execution) for language-instructed outdoor navigation.
https://arxiv.org/abs/2207.04429

**Li, Chen, et al. (2024).** "VLN-Video: Utilizing Driving Videos for Outdoor Vision-and-Language Navigation." AAAI 2024. Outdoor driving videos + auto-generated instructions for VLN pre-training. 2.1% improvement on Touchdown.
https://arxiv.org/abs/2402.03561

**Tian et al. (2024).** "Loc4Plan: Locating Before Planning for Outdoor VLN." ACM MM 2024. Block-aware spatial locating + spatial-aware action planning for outdoor VLN.
https://arxiv.org/abs/2408.05090

### 7.11 Outdoor/Field Robot Dialogue

**Fulton et al. (2022).** "Robot Communication Via Motion: Modalities for Robot-to-Human Communication in the Field." ACM THRI 11(2). RCVM uses "kinemes" (motions with meaning) for field robots in unstructured outdoor environments. Evaluated on AUV, UAV, and ground robot.
https://dl.acm.org/doi/10.1145/3495245

**Love, Andriella, Alenya (2024).** "Towards Explainable Proactive Robot Interactions for Groups in Unstructured Environments." HRI 2024 Companion. Perception and decision-making for proactive interaction in unstructured public spaces.
https://dl.acm.org/doi/10.1145/3610978.3640734

**Sridharan et al. (2023).** "Challenges and Solutions for Autonomous Ground Robot Navigation in Unstructured Outdoor Environments: A Review." Applied Sciences 13(17). Comprehensive review of outdoor robot challenges: terrain analysis, object detection, environmental understanding.
https://www.mdpi.com/2076-3417/13/17/9877

### 7.12 Legged Robot Communication and Navigation

**Tang, Yu, Tan, et al. (2023).** "SayTap: Language to Quadrupedal Locomotion." CoRL 2023. Foot contact patterns bridge NL commands to quadruped locomotion. Language interface for quadrupeds.
https://arxiv.org/abs/2306.07580

**Ding, Zhao, et al. (2024).** "QUAR-VLA: Vision-Language-Action Model for Quadruped Robots." ECCV 2024. Visual info + language instructions → executable actions for quadrupeds including navigation and manipulation.
https://arxiv.org/abs/2312.14457

**Kim, Woo, et al. (2024).** "TOP-Nav: Legged Navigation Integrating Terrain, Obstacle and Proprioception Estimation." 2024. Vision + proprioception synergies for quadruped navigation.
https://arxiv.org/abs/2404.15256

**Rana et al. (2024).** "BehAV: Behavioral Rule Guided Autonomy Using VLMs for Robot Navigation in Outdoor Scenes." ICRA 2025. **Our base system.** VLMs for zero-shot scene understanding + CLIPSeg behavioral cost maps + MPC for outdoor quadruped navigation. 22.49% alignment improvement, 40% higher success rate.
https://arxiv.org/abs/2409.16484

---

## 8. Phase (c) Synthesis — Key Convergence Points

### 8.1 Conformal prediction is the standard for uncertainty-aware robot decisions

Papers [7.2.1], [7.2.2], [7.9.1]-[7.9.6] all converge on conformal prediction as the tool of choice for "knowing when you don't know." Distribution-free coverage guarantees without model assumptions make it ideal for deciding when a behavioral rule conflict requires human intervention. The thread connects KnowNo → Introspective Planning → Formal verification → Multi-robot planning.

**For BehAV:** Apply CP to the behavioral cost distribution along candidate trajectories. If the prediction set of "acceptable trajectories" is empty or contains conflicting-quality options, trigger dialogue.

### 8.2 Deontic logic meets learning for norm conflicts

Papers [7.3.1]-[7.3.5] show that standard LTL/safe RL is insufficient for normative reasoning. Defeasible deontic logic handles:
- **Contrary-to-duty obligations:** "If you violated 'stay on pavement,' you should minimize grass traversal"
- **Norm priority:** "Safety overrides behavioral preferences"
- **Violation counting:** Track cumulative violations, not just binary compliance

**For BehAV:** Formalize behavioral rules as deontic norms. When "stay on pavement" (obligation) conflicts with "pavement ends" (impossibility), the deontic framework produces a contrary-to-duty obligation: "minimize deviation from pavement" or "ask human for guidance."

### 8.3 Proactive explanation outperforms reactive

Paper [7.6.3] (LeMasurier et al., HRI 2024) empirically demonstrates that robots warning about problems BEFORE they occur engender more trust and are perceived as more intelligent. This supports our design: detect behavioral conflicts proactively (from cost map analysis) rather than waiting for execution failure.

### 8.4 Cost map adaptation from human feedback is maturing

Papers [7.5.1]-[7.5.6] show a trajectory from hand-designed → IRL-learned → preference-conditioned → VLM-grounded adaptive cost maps. PACER [7.5.1] and VLM-GroNav [7.5.4] enable dynamic cost map updates that could incorporate human feedback about which behavioral constraints to relax.

**For BehAV:** When the human says "it's OK to cross the grass here," the system should update `self.cost_values` for "grass" from 0.48 (mildly avoid) to 0.7 (mildly prefer) for the duration of the current conflict resolution.

### 8.5 Mixed-initiative dialogue is the emerging paradigm

Papers [7.4.1]-[7.4.5] and [7.10.1] show effective constraint modification requires bidirectional communication where either party can initiate, propose, accept, or reject. This goes beyond "robot asks human for help" to genuine negotiation.

---

## 9. The Gap Our Work Fills

The literature reveals a **clear and specific gap**:

> **No prior work combines behavioral rule conflict detection within cost-map-based navigation with human-robot dialogue for interactive constraint resolution.**

Specifically:
- KnowNo/Introspective Planning [7.2.1-2] → "which action to take" ambiguity, NOT conflicting behavioral constraints
- Constraint relaxation [7.1.1, 7.1.4] → infeasible LTL specs, but no dialogue or natural language
- Norm compliance [7.3.1-3] → formal contrary-to-duty obligations, but not applied to VLM-based cost map navigation
- Mixed-initiative navigation [7.10.1] → social collision avoidance, but not behavioral rule conflicts from decomposed instructions
- BehAV [7.12.4] → behavioral cost maps, but no conflict detection, no communication, no human-in-the-loop

**Our Phase (c) uniquely bridges:**
1. Detection of behavioral rule conflicts within CLIPSeg-generated cost maps and MPC optimization
2. Conformal-prediction-based uncertainty quantification for triggering dialogue
3. LLM-mediated human-robot dialogue for interactive constraint relaxation
4. All within an outdoor quadruped navigation context

### Must-Cite Papers for Phase (c):

| Priority | Paper | Why |
|----------|-------|-----|
| **Critical** | KnowNo (Ren et al., 2023) | Closest work: CP for help-seeking. We extend from task ambiguity to behavioral conflicts |
| **Critical** | Introspective Planning (Liang et al., 2024) | Extends KnowNo with introspection. We apply to navigation-specific conflicts |
| **Critical** | Lahijanian et al. (2015) | Partial satisfaction of specs. Our cost-map conflict = continuous-space analog |
| **Critical** | LeMasurier et al. (2024) | Proactive > reactive explanation. Supports our design |
| **Critical** | Raman et al. (2013) | Explaining unachievable tasks in NL. We extend to behavioral navigation |
| **High** | Neufeld (2023) | Normative Supervisor for contrary-to-duty. Formal grounding for post-violation behavior |
| **High** | Yang et al. (2024) Safety Chip | Constraint enforcement for LLM robots with reprompting |
| **High** | HSAC-LLM (Zhang et al., 2024) | Closest: bidirectional LLM dialogue during navigation |
| **High** | PACER (Mao et al., 2024) | Preference-conditioned cost maps. Integration target |
| **High** | Lindemann et al. (2023) | CP for safe MPC planning. Technical foundation |
| **Medium** | Kim et al. (2024) | Robot decision dilemmas from conflicting constraints |
| **Medium** | AINav (Ma et al., 2025) | Quadruped with LLM help-seeking |
| **Medium** | Charisi et al. (2024) | Dialogue for moral alignment in vehicles |
| **Medium** | LILAC (Cui et al., 2023) | Online NL corrections during execution |
| **Medium** | Guardian (Li et al., 2024) | VLM failure detection |

---

## 10. Evaluation Framework

### 10.1 How Comparable Papers Evaluate

#### BehAV original (Rana et al., ICRA 2025)

| Metric | Definition |
|--------|-----------|
| **Success Rate (SR)** | % of times robot reached goal while avoiding collisions and following behavioral rules |
| **Fréchet Distance (FD)** | Distance between method's trajectory and human-teleoperated trajectory (lower = more human-like) |
| **Behavior Following Accuracy (BFA)** | % of robot's path length that adhered to behavioral rules / total path length |
| **Pixel Error** | Average distance error between ground truth and predicted landmark pixel coordinates |
| **F-score** | Accuracy of landmark location within ground truth region |
| **Average Goal Heading Error** | Angular deviation (radians) between goal line-of-sight and robot heading |

- **Baselines:** CoNVOI, GA-Nav, ViNT, NoMAD, DWA (ablation)
- **Environments:** 5 real-world outdoor scenarios (sidewalks, crosswalks, multi-terrain, vegetation, stairs)
- **Trials:** Minimum 10 per scenario
- **Results:** SR 70-90%, BFA 76-89%, FD 0.99-5.47m

#### KnowNo (Ren et al., CoRL 2023)

| Metric | Definition |
|--------|-----------|
| **Task Success Rate** | Probability the correct action is in the prediction set |
| **Help Rate (HR)** | Frequency the robot asks humans for assistance (lower = more autonomous) |
| **Prediction Set Size** | Average number of options in the conformal prediction set (1 = certain) |

- **Threshold:** Conformal prediction calibrated for 85% target success rate
- **Trigger:** If prediction set has >1 option → ask human
- **Domains:** Mobile manipulation (15 instructions), tabletop rearrangement (10 combos), bimanual manipulation (10 instructions)
- **Task types:** Single-label, multi-label, spatially-ambiguous, unsafe, Winograd

#### Introspective Planning (Liang et al., NeurIPS 2024)

| Metric | Definition |
|--------|-----------|
| **SR** | Success Rate |
| **HR** | Help Rate (lower = less overasking) |
| **OAR** | Over-Ask Rate — % of queries where human help was unnecessary |
| **OSR** | Over-Step Rate — % of times robot acted when it should have asked |
| **UR** | Unsafe Rate — % of actions that violated safety constraints |
| **ESR** | Exact Set Rate — % of prediction sets that exactly match the correct option set |
| **NCR** | Non-Compliance Contamination Rate — prediction sets containing non-compliant actions |
| **UCR** | Unsafe Contamination Rate — prediction sets containing unsafe actions |

- **Baselines:** KnowNo, Retrieval-Q-CoT
- **Benchmark:** New "Safe Mobile Manipulation" benchmark (augments KnowNo tasks with safety-critical scenarios)
- **Key improvement over KnowNo:** Highest ESR, lowest NCR and UCR, avoids overasking

#### VLM-Social-Nav (Raj et al., RA-L 2024)

| Metric | Definition |
|--------|-----------|
| **Success Rate** | Robot reaches goal respecting social norms |
| **Collision Rate** | Manual intervention needed to avoid imminent collision with humans |
| **User Study Score** | Average participant rating of social compliance |

- **Baselines:** DWA, Behavior Cloning (trained on SCAND dataset)
- **Scenarios:** 4 social navigation scenarios (frontal approach + gesture, etc.)
- **Results:** +27.38% SR improvement, +19.05% collision reduction over baselines

### 10.2 Proposed Evaluation for BehAV-Talk

#### A. Scenario Benchmark (BehAV-Conflict)

A purpose-built benchmark of **24 scenarios** (8 conflict types × 3 difficulty levels) in Go2 Gazebo simulation:

| # | Conflict Type | Example | Ground Truth |
|---|--------------|---------|-------------|
| 1 | **Terrain disappears** | "Stay on pavement" but pavement ends | Must ask: cross grass or wait |
| 2 | **Contradictory rules** | "Stay on path" + "Avoid crowd on path" | Must ask: which rule has priority |
| 3 | **Obstacle on preferred path** | Preferred path blocked by obstacle | Must ask: alternative route or wait |
| 4 | **All-bad terrain** | All visible terrain is "avoid" category | Must ask: proceed anyway or stop |
| 5 | **Priority ambiguity** | "Stop at gesture" + "Go quickly to goal" | Must ask: which overrides |
| 6 | **Shortcut vs. compliance** | Shortcut through forbidden terrain available | Should NOT ask (follow rules) |
| 7 | **Landmark not found** | Target landmark not visible in any direction | Must ask: description, rotate, skip |
| 8 | **Temporal rule change** | Rules valid initially, become invalid mid-execution | Must detect change and ask |

Difficulty levels per type:
- **Easy:** Single clear conflict, obvious trigger point
- **Medium:** Conflict requires 2-3 inference steps to detect
- **Hard:** Subtle conflict (e.g., CLIPSeg confidence gradually degrades)

Each scenario has **ground truth annotations**:
- Whether a conflict exists (binary)
- Type of conflict
- Optimal timing to trigger dialogue (proactive window)
- Correct set of question options
- Expected outcome after human response

#### B. Baselines (5 configurations)

| ID | Baseline | Description |
|----|----------|-------------|
| B1 | **BehAV-original** | No dialogue, no conflict detection (current planner) |
| B2 | **Always-ask** | Queries human at every planning cycle (upper bound on help rate) |
| B3 | **Threshold-ask** | Queries when `max_behav_cost / 255 > τ` (fixed threshold τ=0.6) |
| B4 | **KnowNo-adapted** | Conformal prediction on discrete action set {proceed, stop, turn-left, turn-right, ask-human} |
| B5 | **BehAV-Talk (ours)** | Full system: conflict detection + conformal prediction + contextual LLM dialogue |

#### C. Metrics

Three groups aligned with the three dimensions of contribution:

**Group 1: Navigation Performance** (extends BehAV metrics)

| Metric | Definition | Source |
|--------|-----------|--------|
| **SR** (Success Rate) | % missions where robot reaches goal | BehAV |
| **BFA** (Behavior Following Accuracy) | % path length adhering to behavioral rules / total path | BehAV |
| **FD** (Fréchet Distance) | Trajectory similarity to human-operated reference path | BehAV |
| **MT** (Mission Time) | Total time from start to goal (seconds) | Standard |
| **PE** (Path Efficiency) | `distance_optimal / distance_actual` | Standard |
| **WRV** (Weighted Rule Violation) | `Σ(violation_duration × severity_weight)` normalized by path length | **New** |

WRV severity weights: crossing grass=0.3, crossing vegetation=0.5, ignoring stop gesture=1.0, entering road=1.0

**Group 2: Conflict Detection & Dialogue Quality** (extends KnowNo/IntroPlan metrics)

| Metric | Definition | Source |
|--------|-----------|--------|
| **HR** (Help Rate) | % planning cycles requiring human query | KnowNo |
| **Precision-C** | % of triggered dialogues that correspond to a real conflict | IntroPlan (ESR analog) |
| **Recall-C** | % of real conflicts that triggered a dialogue | **New** |
| **F1-C** | Harmonic mean of Precision-C and Recall-C | **New** |
| **OAR** (Over-Ask Rate) | % of queries where help was unnecessary | IntroPlan |
| **OSR** (Over-Step Rate) | % of conflicts where robot acted without asking | IntroPlan |
| **QT** (Query Timing) | Time between conflict onset and query trigger (seconds, lower = more proactive) | **New** |
| **UCR** (Unsafe Contamination Rate) | % of prediction sets containing unsafe actions | IntroPlan |

**Group 3: Subjective (User Study only)**

| Metric | Instrument | Measures |
|--------|-----------|----------|
| **Task Load** | NASA-TLX | Human cognitive load from robot queries |
| **Trust** | Trust Perception Scale (Schaefer 2016) | Trust in robot's decisions and communication |
| **Usability** | System Usability Scale (SUS) | Overall system usability |
| **Explanation Quality** | 5-point Likert scale | Were robot's questions clear, relevant, well-timed? |
| **Social Compliance** | 5-point Likert scale | Per VLM-Social-Nav protocol |

#### D. Evaluation Protocol

**Phase 1: Automated Simulation (no human in the loop)**

1. Run all 24 scenarios × 5 baselines × 10 repetitions = **1,200 trials**
2. Human responses simulated by a **Wizard-of-Oz oracle script** that:
   - Reads ground truth for each scenario
   - Responds optimally to robot queries (simulates perfect human)
   - Introduces response delay (2-5s uniform random) to simulate real interaction
3. Collect: SR, BFA, FD, MT, PE, WRV, HR, Precision-C, Recall-C, F1-C, OAR, OSR, QT, UCR
4. Statistical analysis: Wilcoxon signed-rank tests (paired, non-parametric) with Holm-Bonferroni correction

**Phase 2: Automated with noisy human model**

Same 1,200 trials, but the oracle introduces:
- 10% probability of suboptimal response (choose second-best option)
- 5% probability of non-response (timeout, robot must decide alone)
- Tests robustness of the system to imperfect human partners

**Phase 3: User Study (N=20, within-subjects)**

1. Each participant completes 8 scenarios (one per conflict type, medium difficulty)
2. Three conditions (counterbalanced Latin square): BehAV-original, Threshold-ask, BehAV-Talk
3. Gazebo GUI via noVNC — participant sees robot's camera view + receives text queries
4. Collect: All Group 1 + Group 2 metrics + all Group 3 subjective scales
5. Post-study interview: qualitative feedback on dialogue utility
6. IRB approval required

#### E. Expected Results & Hypotheses

| Hypothesis | Metric | Expected outcome |
|-----------|--------|-----------------|
| H1: BehAV-Talk improves task success in conflict scenarios | SR | BehAV-Talk > BehAV-original by 15-25% |
| H2: BehAV-Talk maintains behavioral compliance | BFA | BehAV-Talk ≥ BehAV-original |
| H3: BehAV-Talk asks less than Always-ask | HR | BehAV-Talk << Always-ask (by 60-80%) |
| H4: BehAV-Talk detects conflicts accurately | F1-C | BehAV-Talk > Threshold-ask and KnowNo-adapted |
| H5: BehAV-Talk asks proactively | QT | BehAV-Talk < 0 (asks BEFORE conflict manifests as failure) |
| H6: Human cognitive load is acceptable | NASA-TLX | BehAV-Talk < Always-ask |
| H7: BehAV-Talk is robust to noisy humans | SR (Phase 2) | BehAV-Talk degrades gracefully (<10% SR drop) |

#### F. Key Ablation Studies

| Ablation | What it tests |
|----------|--------------|
| Remove conformal prediction, keep threshold | Value of statistical guarantees vs fixed threshold |
| Remove LLM dialogue, keep binary ask/don't-ask | Value of contextual questions vs "help me" button |
| Remove proactive detection, only detect after optimization failure | Value of proactive vs reactive conflict detection |
| Vary CP target coverage (80%, 85%, 90%, 95%) | Tradeoff curve: SR vs HR at different confidence levels |
| Vary number of CLIPSeg prompts (2, 4, 6, 8) | Effect of behavioral rule complexity on conflict frequency |

#### G. Comparison Table Design

The paper should include a table like:

| Method | SR↑ | BFA↑ | FD↓ | HR↓ | F1-C↑ | OAR↓ | WRV↓ | MT |
|--------|-----|------|-----|-----|-------|------|------|-----|
| BehAV-original | X | X | X | 0% | N/A | N/A | X | X |
| Always-ask | X | X | X | 100% | N/A | X | X | X |
| Threshold-ask | X | X | X | X | X | X | X | X |
| KnowNo-adapted | X | X | X | X | X | X | X | X |
| **BehAV-Talk** | **X** | **X** | **X** | **X** | **X** | **X** | **X** | **X** |

With arrows indicating desired direction (↑ higher is better, ↓ lower is better).

### 10.3 Why This Evaluation Design Is Strong

1. **Extends BehAV's own metrics** (SR, BFA, FD) — enables direct comparison with original paper
2. **Adopts IntroPlan's dialogue metrics** (HR, OAR, OSR, UCR) — positions against SOTA help-seeking
3. **Introduces novel conflict-specific metrics** (Precision-C, Recall-C, F1-C, QT, WRV) — measures our unique contribution
4. **Three evaluation phases** — automated (scale), noisy (robustness), user study (ecological validity)
5. **Ablation studies** — isolate each component's contribution
6. **Statistical rigor** — non-parametric tests, multiple comparison correction, sufficient sample sizes
