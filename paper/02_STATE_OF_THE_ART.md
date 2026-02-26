# 2. State of the Art (2024-2026)

## 2.1 Human-Robot Dialogue in Navigation

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

## 2.2 VLM/LLM-based Robot Navigation

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

## 2.3 Scene Graph-Based Navigation

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

## 2.4 Behavioral / Social Navigation

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

## 2.5 Instruction Decomposition Improvements

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

## 2.6 Landmark Detection / Segmentation Evolution

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
