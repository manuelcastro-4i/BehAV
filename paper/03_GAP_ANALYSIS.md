# 3. Research Gap Analysis & Paper Proposal

## 3.1 What Exists

- "Ask for help" with conformal prediction (KnowNo, Introspective Planning) — indoor manipulation/navigation
- Bidirectional dialogue for navigation (DRAGON, ORION, DriVLMe) — assistive/driving
- VLM-based social navigation cost maps (VLM-Social-Nav) — indoor social norms
- End-to-end VLM navigation (NaVid, NaVILA) — no behavioral rules

## 3.2 What Does NOT Exist (Our Gap)

- **Dialogue for behavioral rule conflicts in outdoor navigation**
- **Conformal prediction applied to behavioral cost map ambiguity**
- **Robot communicating about terrain/behavioral constraint violations**
- **Interactive relaxation of behavioral rules with human oversight**
- **Outdoor legged robot that explains its behavioral navigation decisions**

---

## 4. Paper Proposal

### Title Options
- "Interactive Behavioral Navigation: Enabling Quadruped Robots to Communicate When Behavioral Rules Conflict"
- "BehAV-Talk: Human-Robot Dialogue for Behavioral Navigation Under Uncertainty"
- "When Rules Break: Interactive Behavioral Navigation with Human-in-the-Loop Conflict Resolution"

### Proposed Contributions
1. **Behavioral dialogue framework**: When/how a robot should communicate during behavioral navigation (3 phases: obstacles, paths, ambiguity)
2. **Behavioral conflict detection**: Method to detect unsatisfiable/ambiguous behavioral constraints using LLM introspection
3. **Perception upgrades**: Grounding DINO + SAM2 for real-time landmark tracking + structured instruction decomposition
4. **Evaluation**: Scenarios in Go2 Gazebo simulation with metrics: success rate, human queries, mission time, rule violations

### Differentiator vs. State of the Art
KnowNo/Introspective Planning handle "which object to grasp" ambiguity. We handle "which behavioral rule to violate when there's no valid path" — fundamentally different domain.

The modular pipeline (vs. end-to-end) is a feature, not a bug: interpretability enables communication. An end-to-end model can't explain WHY it chose a path.

### Risk
Field moving toward end-to-end (NaVILA, NaVid). Counter: emphasize interpretability + communicability as the research contribution, not the pipeline architecture itself.

### Three Dialogue Phases

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

### Technical Improvements Needed

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
