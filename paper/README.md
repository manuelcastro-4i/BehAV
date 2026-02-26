# Paper Research: Interactive Behavioral Navigation

Research materials for the BehAV-Talk paper, organized by topic.

## Files

| File | Content | Lines |
|------|---------|-------|
| [01_CODEBASE_ANALYSIS.md](01_CODEBASE_ANALYSIS.md) | BehAV current state: instruction decomposition, landmark tracking, planner analysis | ~70 |
| [02_STATE_OF_THE_ART.md](02_STATE_OF_THE_ART.md) | 30+ papers: dialogue, VLM nav, scene graphs, social nav, segmentation | ~210 |
| [03_GAP_ANALYSIS.md](03_GAP_ANALYSIS.md) | Research gap + paper proposal (3 phases, title options, technical improvements) | ~100 |
| [04_PHASE_C_CODE_ANALYSIS.md](04_PHASE_C_CODE_ANALYSIS.md) | Deep dive: conflict mechanics, inverted costs bug, 5 scenarios, 5 injection points, missing mechanisms | ~200 |
| [05_PHASE_C_LITERATURE.md](05_PHASE_C_LITERATURE.md) | 55+ papers across 12 sub-topics for phase (c) | ~250 |
| [06_SYNTHESIS.md](06_SYNTHESIS.md) | Convergence points, gap definition, must-cite table | ~80 |
| [07_EVALUATION.md](07_EVALUATION.md) | Full evaluation framework: metrics, baselines, scenarios, protocol, hypotheses, ablations | ~220 |

## Quick Reference

- **Full monolithic version:** [../PAPER_RESEARCH.md](../PAPER_RESEARCH.md) (1080 lines, all content)
- **Paper title candidates:**
  - "Interactive Behavioral Navigation: Enabling Quadruped Robots to Communicate When Behavioral Rules Conflict"
  - "BehAV-Talk: Human-Robot Dialogue for Behavioral Navigation Under Uncertainty"
  - "When Rules Break: Interactive Behavioral Navigation with Human-in-the-Loop Conflict Resolution"
- **Core gap:** No prior work combines behavioral rule conflict detection within cost-map-based navigation with human-robot dialogue for interactive constraint resolution
- **Total references:** 80+ papers
