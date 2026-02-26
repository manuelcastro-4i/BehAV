# 1. BehAV Codebase Analysis — Current State

## 1.1 Instruction Decomposition (`instruction-decomposition/GPT4o/instruction-to-behavioral-costs.py`)

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

## 1.2 Landmark Tracking (`landmark-tracking/landmark_detector_ROS2.py`, `landmarkdetector_fastsam.py`)

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

## 1.3 Planning (`planning/behav-planner.py`)

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
