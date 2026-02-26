# 10. Evaluation Framework

## 10.1 How Comparable Papers Evaluate

### BehAV original (Rana et al., ICRA 2025)

| Metric | Definition |
|--------|-----------|
| **Success Rate (SR)** | % of times robot reached goal while avoiding collisions and following behavioral rules |
| **Frechet Distance (FD)** | Distance between method's trajectory and human-teleoperated trajectory (lower = more human-like) |
| **Behavior Following Accuracy (BFA)** | % of robot's path length that adhered to behavioral rules / total path length |
| **Pixel Error** | Average distance error between ground truth and predicted landmark pixel coordinates |
| **F-score** | Accuracy of landmark location within ground truth region |
| **Average Goal Heading Error** | Angular deviation (radians) between goal line-of-sight and robot heading |

- **Baselines:** CoNVOI, GA-Nav, ViNT, NoMAD, DWA (ablation)
- **Environments:** 5 real-world outdoor scenarios (sidewalks, crosswalks, multi-terrain, vegetation, stairs)
- **Trials:** Minimum 10 per scenario
- **Results:** SR 70-90%, BFA 76-89%, FD 0.99-5.47m

### KnowNo (Ren et al., CoRL 2023)

| Metric | Definition |
|--------|-----------|
| **Task Success Rate** | Probability the correct action is in the prediction set |
| **Help Rate (HR)** | Frequency the robot asks humans for assistance (lower = more autonomous) |
| **Prediction Set Size** | Average number of options in the conformal prediction set (1 = certain) |

- **Threshold:** Conformal prediction calibrated for 85% target success rate
- **Trigger:** If prediction set has >1 option → ask human
- **Domains:** Mobile manipulation (15 instructions), tabletop rearrangement (10 combos), bimanual manipulation (10 instructions)
- **Task types:** Single-label, multi-label, spatially-ambiguous, unsafe, Winograd

### Introspective Planning (Liang et al., NeurIPS 2024)

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

### VLM-Social-Nav (Raj et al., RA-L 2024)

| Metric | Definition |
|--------|-----------|
| **Success Rate** | Robot reaches goal respecting social norms |
| **Collision Rate** | Manual intervention needed to avoid imminent collision with humans |
| **User Study Score** | Average participant rating of social compliance |

- **Baselines:** DWA, Behavior Cloning (trained on SCAND dataset)
- **Scenarios:** 4 social navigation scenarios (frontal approach + gesture, etc.)
- **Results:** +27.38% SR improvement, +19.05% collision reduction over baselines

---

## 10.2 Proposed Evaluation for BehAV-Talk

### A. Scenario Benchmark (BehAV-Conflict)

A purpose-built benchmark of **24 scenarios** (8 conflict types x 3 difficulty levels) in Go2 Gazebo simulation:

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

### B. Baselines (5 configurations)

| ID | Baseline | Description |
|----|----------|-------------|
| B1 | **BehAV-original** | No dialogue, no conflict detection (current planner) |
| B2 | **Always-ask** | Queries human at every planning cycle (upper bound on help rate) |
| B3 | **Threshold-ask** | Queries when `max_behav_cost / 255 > t` (fixed threshold t=0.6) |
| B4 | **KnowNo-adapted** | Conformal prediction on discrete action set {proceed, stop, turn-left, turn-right, ask-human} |
| B5 | **BehAV-Talk (ours)** | Full system: conflict detection + conformal prediction + contextual LLM dialogue |

### C. Metrics

Three groups aligned with the three dimensions of contribution:

**Group 1: Navigation Performance** (extends BehAV metrics)

| Metric | Definition | Source |
|--------|-----------|--------|
| **SR** (Success Rate) | % missions where robot reaches goal | BehAV |
| **BFA** (Behavior Following Accuracy) | % path length adhering to behavioral rules / total path | BehAV |
| **FD** (Frechet Distance) | Trajectory similarity to human-operated reference path | BehAV |
| **MT** (Mission Time) | Total time from start to goal (seconds) | Standard |
| **PE** (Path Efficiency) | `distance_optimal / distance_actual` | Standard |
| **WRV** (Weighted Rule Violation) | `sum(violation_duration x severity_weight)` normalized by path length | **New** |

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

### D. Evaluation Protocol

**Phase 1: Automated Simulation (no human in the loop)**

1. Run all 24 scenarios x 5 baselines x 10 repetitions = **1,200 trials**
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

### E. Expected Results & Hypotheses

| Hypothesis | Metric | Expected outcome |
|-----------|--------|-----------------|
| H1: BehAV-Talk improves task success in conflict scenarios | SR | BehAV-Talk > BehAV-original by 15-25% |
| H2: BehAV-Talk maintains behavioral compliance | BFA | BehAV-Talk >= BehAV-original |
| H3: BehAV-Talk asks less than Always-ask | HR | BehAV-Talk << Always-ask (by 60-80%) |
| H4: BehAV-Talk detects conflicts accurately | F1-C | BehAV-Talk > Threshold-ask and KnowNo-adapted |
| H5: BehAV-Talk asks proactively | QT | BehAV-Talk < 0 (asks BEFORE conflict manifests as failure) |
| H6: Human cognitive load is acceptable | NASA-TLX | BehAV-Talk < Always-ask |
| H7: BehAV-Talk is robust to noisy humans | SR (Phase 2) | BehAV-Talk degrades gracefully (<10% SR drop) |

### F. Key Ablation Studies

| Ablation | What it tests |
|----------|--------------|
| Remove conformal prediction, keep threshold | Value of statistical guarantees vs fixed threshold |
| Remove LLM dialogue, keep binary ask/don't-ask | Value of contextual questions vs "help me" button |
| Remove proactive detection, only detect after optimization failure | Value of proactive vs reactive conflict detection |
| Vary CP target coverage (80%, 85%, 90%, 95%) | Tradeoff curve: SR vs HR at different confidence levels |
| Vary number of CLIPSeg prompts (2, 4, 6, 8) | Effect of behavioral rule complexity on conflict frequency |

### G. Comparison Table Design

The paper should include a table like:

| Method | SR^ | BFA^ | FD | HR | F1-C^ | OAR | WRV | MT |
|--------|-----|------|-----|-----|-------|------|------|-----|
| BehAV-original | X | X | X | 0% | N/A | N/A | X | X |
| Always-ask | X | X | X | 100% | N/A | X | X | X |
| Threshold-ask | X | X | X | X | X | X | X | X |
| KnowNo-adapted | X | X | X | X | X | X | X | X |
| **BehAV-Talk** | **X** | **X** | **X** | **X** | **X** | **X** | **X** | **X** |

With arrows indicating desired direction (^ higher is better, lower is better for the rest).

## 10.3 Why This Evaluation Design Is Strong

1. **Extends BehAV's own metrics** (SR, BFA, FD) — enables direct comparison with original paper
2. **Adopts IntroPlan's dialogue metrics** (HR, OAR, OSR, UCR) — positions against SOTA help-seeking
3. **Introduces novel conflict-specific metrics** (Precision-C, Recall-C, F1-C, QT, WRV) — measures our unique contribution
4. **Three evaluation phases** — automated (scale), noisy (robustness), user study (ecological validity)
5. **Ablation studies** — isolate each component's contribution
6. **Statistical rigor** — non-parametric tests, multiple comparison correction, sufficient sample sizes
