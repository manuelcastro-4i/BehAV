# 7. Phase (c) Deep Dive — Expanded Literature Review (55+ Papers)

## 7.1 Constraint Conflict/Violation Detection in Robot Planning

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

## 7.2 LLM/VLM Introspection and Uncertainty Quantification

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

## 7.3 Norm Compliance and Violation in Autonomous Agents

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

## 7.4 Human-in-the-Loop Constraint Modification

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

## 7.5 Cost Map Adaptation and Dynamic Replanning

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

## 7.6 Explainable Robot Navigation

**Halilovic (2025).** "Explainable Robot Navigation." AAAI 2025 Doctoral Consortium. HiXRoN: Hierarchical eXplainable Robot Navigation combining ML with symbolic reasoning for multimodal, personalized explanations. **Directly relevant to explaining behavioral rule conflicts.**
https://ojs.aaai.org/index.php/AAAI/article/view/35208

**Halilovic, Krivic, Canal (2024).** "Towards Probabilistic Planning of Explanations for Robot Navigation." arXiv 2024. RDDL-based framework for planning explanations tailored to user preferences (textual vs. visual, rich vs. poor, local vs. global).
https://arxiv.org/abs/2411.05022

**LeMasurier, Gautam, Han, Crandall, Yanco (2024).** "Reactive or Proactive? How Robots Should Explain Failures." HRI 2024. **Proactive explanations (predicting issues in advance) foster higher trust and perceived intelligence than reactive ones.** Supports our design for proactive conflict communication.
https://dl.acm.org/doi/10.1145/3610977.3634963

**Tagliamonte, Maccaline (2024).** "A Generalizable Architecture for Explaining Robot Failures." UMass Lowell. Template-based NLG for robot failure explanations, adaptable to different platforms and failure types.

**Halilovic, Krivic (2024).** "Towards a Holistic Framework for Explainable Robot Navigation." Springer LNCS. Unified framework integrating explanation modalities, temporal strategies, and personalization.
https://link.springer.com/chapter/10.1007/978-3-031-55000-3_15

## 7.7 Failure Recovery and Help-Seeking in Robotics

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

## 7.8 Ethical Decision-Making in Autonomous Systems

**Awad, Dsouza, Kim, et al. (2018).** "The Moral Machine Experiment." Nature 563, 59-64. 40 million ethical decisions from millions of people about AV dilemmas. Reveals global moral preferences and cultural variations. Foundational for ethical trade-offs in autonomous systems.
https://www.nature.com/articles/s41586-018-0637-6

**Chiu, Jiang, et al. (2024).** "DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life." ICLR 2025. 1,360 everyday moral dilemmas; finds LLMs have implicit value priorities that resist steering via system prompts. Relevant to understanding LLM-based planners handling behavioral conflicts.
https://arxiv.org/abs/2410.02683

**Charisi, Liem, Gomez (2024).** "Human-Robot Dialogue that Elicits Alignment of Moral Principles For Driverless Vehicles." HRI 2024 Companion. Dialogue system for eliciting ethical preferences from users to align AV behavior. **Directly relevant to using dialogue to resolve behavioral rule conflicts.**
https://dl.acm.org/doi/10.1145/3610978.3640641

**Geisslinger, Poszler, Betz, Lienkamp (2024).** "Ethical Decision-Making for Self-Driving Vehicles." Science and Engineering Ethics. Model for ethical decision-making with explicit moral attribute definitions, bridging abstract principles and implementable algorithms.
https://link.springer.com/article/10.1007/s11948-024-00513-0

## 7.9 Conformal Prediction in Robotics

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

## 7.10 Mixed-Initiative Interaction for Navigation

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

## 7.11 Outdoor/Field Robot Dialogue

**Fulton et al. (2022).** "Robot Communication Via Motion: Modalities for Robot-to-Human Communication in the Field." ACM THRI 11(2). RCVM uses "kinemes" (motions with meaning) for field robots in unstructured outdoor environments. Evaluated on AUV, UAV, and ground robot.
https://dl.acm.org/doi/10.1145/3495245

**Love, Andriella, Alenya (2024).** "Towards Explainable Proactive Robot Interactions for Groups in Unstructured Environments." HRI 2024 Companion. Perception and decision-making for proactive interaction in unstructured public spaces.
https://dl.acm.org/doi/10.1145/3610978.3640734

**Sridharan et al. (2023).** "Challenges and Solutions for Autonomous Ground Robot Navigation in Unstructured Outdoor Environments: A Review." Applied Sciences 13(17). Comprehensive review of outdoor robot challenges: terrain analysis, object detection, environmental understanding.
https://www.mdpi.com/2076-3417/13/17/9877

## 7.12 Legged Robot Communication and Navigation

**Tang, Yu, Tan, et al. (2023).** "SayTap: Language to Quadrupedal Locomotion." CoRL 2023. Foot contact patterns bridge NL commands to quadruped locomotion. Language interface for quadrupeds.
https://arxiv.org/abs/2306.07580

**Ding, Zhao, et al. (2024).** "QUAR-VLA: Vision-Language-Action Model for Quadruped Robots." ECCV 2024. Visual info + language instructions → executable actions for quadrupeds including navigation and manipulation.
https://arxiv.org/abs/2312.14457

**Kim, Woo, et al. (2024).** "TOP-Nav: Legged Navigation Integrating Terrain, Obstacle and Proprioception Estimation." 2024. Vision + proprioception synergies for quadruped navigation.
https://arxiv.org/abs/2404.15256

**Rana et al. (2024).** "BehAV: Behavioral Rule Guided Autonomy Using VLMs for Robot Navigation in Outdoor Scenes." ICRA 2025. **Our base system.** VLMs for zero-shot scene understanding + CLIPSeg behavioral cost maps + MPC for outdoor quadruped navigation. 22.49% alignment improvement, 40% higher success rate.
https://arxiv.org/abs/2409.16484
