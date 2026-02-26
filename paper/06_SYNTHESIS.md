# 8. Phase (c) Synthesis — Key Convergence Points

## 8.1 Conformal prediction is the standard for uncertainty-aware robot decisions

Papers [7.2.1], [7.2.2], [7.9.1]-[7.9.6] all converge on conformal prediction as the tool of choice for "knowing when you don't know." Distribution-free coverage guarantees without model assumptions make it ideal for deciding when a behavioral rule conflict requires human intervention. The thread connects KnowNo → Introspective Planning → Formal verification → Multi-robot planning.

**For BehAV:** Apply CP to the behavioral cost distribution along candidate trajectories. If the prediction set of "acceptable trajectories" is empty or contains conflicting-quality options, trigger dialogue.

## 8.2 Deontic logic meets learning for norm conflicts

Papers [7.3.1]-[7.3.5] show that standard LTL/safe RL is insufficient for normative reasoning. Defeasible deontic logic handles:
- **Contrary-to-duty obligations:** "If you violated 'stay on pavement,' you should minimize grass traversal"
- **Norm priority:** "Safety overrides behavioral preferences"
- **Violation counting:** Track cumulative violations, not just binary compliance

**For BehAV:** Formalize behavioral rules as deontic norms. When "stay on pavement" (obligation) conflicts with "pavement ends" (impossibility), the deontic framework produces a contrary-to-duty obligation: "minimize deviation from pavement" or "ask human for guidance."

## 8.3 Proactive explanation outperforms reactive

Paper [7.6.3] (LeMasurier et al., HRI 2024) empirically demonstrates that robots warning about problems BEFORE they occur engender more trust and are perceived as more intelligent. This supports our design: detect behavioral conflicts proactively (from cost map analysis) rather than waiting for execution failure.

## 8.4 Cost map adaptation from human feedback is maturing

Papers [7.5.1]-[7.5.6] show a trajectory from hand-designed → IRL-learned → preference-conditioned → VLM-grounded adaptive cost maps. PACER [7.5.1] and VLM-GroNav [7.5.4] enable dynamic cost map updates that could incorporate human feedback about which behavioral constraints to relax.

**For BehAV:** When the human says "it's OK to cross the grass here," the system should update `self.cost_values` for "grass" from 0.48 (mildly avoid) to 0.7 (mildly prefer) for the duration of the current conflict resolution.

## 8.5 Mixed-initiative dialogue is the emerging paradigm

Papers [7.4.1]-[7.4.5] and [7.10.1] show effective constraint modification requires bidirectional communication where either party can initiate, propose, accept, or reject. This goes beyond "robot asks human for help" to genuine negotiation.

---

# 9. The Gap Our Work Fills

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

## Must-Cite Papers for Phase (c)

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
