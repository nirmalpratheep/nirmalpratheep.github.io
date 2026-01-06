# Nirmal Pratheep Natarajan
**Design Automation Software Engineer @ AMD**  
[nirmalpratheep@gmail.com](mailto:nirmalpratheep@gmail.com) | [GitHub](https://github.com/nirmalpratheep) | [LinkedIn](https://linkedin.com/in/nirmalpratheep)

---

**12+ years** in software engineering at AMD, specializing in **AI/ML-driven solutions** from research to production. Currently focused on **LLM training pipelines** (SFT, GRPO, RLHF), **Agentic AI systems**, and **distributed training infrastructure**. Passionate about **AI Alignment**, **Deep Reinforcement Learning**, and scaling **large-scale model training**.

## 🔥 Highlights

- **Alignment & RL** — Post-LLM training (**Model**: Qwen 2.5 Math 1.5B) on **Math Dataset (Hendrycks 2021)**; achieved **14.2× zero-shot accuracy** (Base 2% → SFT 26% → GRPO **40.46%**)
- **AI in EDA** — Deep RL for FloorPlan optimization, LLM-based triage decisions
- **Distributed Systems** — Ray, LSF Farm, Client/Server concurrent capture, **3× throughput**
- **Leadership** — Technical Lead of **10+ engineers** across global sites
- **AI/ML** — Pre-trained **135M LLM**, ResNet-50 **77.4% Top-1** on ImageNet

## 📚 Research & Publications

- **Deep RL FloorPlan** — GTAC'25 & SPS Tech Conference *(Finalist, arXiv pending)*
- **ML Delay Prediction** — GTAC'22 AMD Tech Conference *(Finalist)*
- **Adaptive OFDM Pilots** — IEEE WAMICON 2009

## 💼 Industry Experience

### Deep RL for Directive Optimization
- Researched and developed **Environment/RL models** to solve directive optimization for EDA tool performance
- Implemented **GIN (Graph Isomorphism Network)** for feature extraction from 1 to 15 million node graph netlists
- Developed **token-based feature collection** to avoid normalization problems between train/test splits and eliminate feature variation distribution issues
- Employed **Grid, ASHA, and PBT** hyperparameter search for automated optimization
- Built **Ray-based distributed training** infrastructure for scalable RL model training across multiple nodes
- Achieved 2% improvement in Placement quality while replacing manual tuning

### Tool Chain Optimization
- Technical Team Lead of 10+ members; built tool profiling, linters, debugging tools, and dashboards
- Automatic YAML semantic verifier auto code generation tool to validate syntax and contents for HW/SW assumptions
- Developed tool that splits entire device using divide-and-conquer approach, parallel processing each division of the chip through **LSF Farm** and merging results to handle/manage both compute/memory and overall throughput; enabled handling of **20x larger chip versions** compared to initial chip size

### Simulation Delay Capture Tools for 10nm/7nm/2nm FPGAs
- Senior Team Member in design/development from scratch - mentor and guide 10+ engineers
- Designed Client/Server model using **Boost Asio** and **Google Protobuf**; enabled **concurrent multi-capture** with 3× throughput improvement
- Built high-performance pattern recognition processing 3.5B datapoints via clustering and distributed computation
- Developed graph compression techniques (1 Billion Instance Paths to 500K patterns) with Python analytics and visualization tools

### AI/ML Projects
- Developing **Agentic AI framework** with **LLM-driven decision making** for autonomous triage and reruns
- Developed **delay prediction algorithms** using **ML techniques**
- Built **feature engineering pipeline** using **GNN models** and graph clustering for EDA router tool Design complexity prediction
- Part of the team developing **regression framework** with **model monitoring** and **automated fine-tuning** when performance degrades; handles both **model drift** and **data drift**

## 🚀 AI/ML Projects

### RL Projects
**[Alignment_and_Reasoning_RL](https://github.com/nirmalpratheep/Alignment_and_Reasoning_RL)**
- Built complete **LLM alignment pipeline** for **Qwen 2.5 Math 1.5B** on **Math Dataset (Hendrycks 2021)**: Baseline → **SFT** → **GRPO RL** achieving **14.2× zero-shot accuracy** (2.84% → 40.46%)
- Implemented **TRL GRPOTrainer** with **vLLM colocate mode** for high-throughput RL rollouts; achieved **96.72% format accuracy**
- Designed **Optuna + ASHA** hyperparameter optimization with **dual-GPU pipeline** (training on GPU 0, vLLM eval on GPU 1)
- **W&B Experiments**: [HPO](https://wandb.ai/nirmalpratheep-self/math-sft-optuna-asha) | [SFT](https://wandb.ai/nirmalpratheep-self/math-sft) | [GRPO](https://wandb.ai/nirmalpratheep-self/math-grpo-trl)

**[RL-CarNavigationAgent](https://github.com/nirmalpratheep/RL-CarNavigationAgent)**
- Tuned **RL hyperparameters** (Gamma 0.95, Tau 0.005) using **TensorBoard**; achieved **stable learning (< 1.0 KL Divergence)**
- Optimized **DQN agent** physics parameters; visualized **Average Reward** trends and gradient norms

### LLM Projects
**[SmolLMv2-PreTrain](https://github.com/nirmalpratheep/SmolLMv2-PreTrain)**
- Implemented **SmolLM v2 pre-training** (**135M parameters**) on **FineWeb-Edu dataset** with **Flash Attention**
- Achieved **~40k tokens/sec throughput** using **Mixed Precision (BF16)**; reduced loss from 11.6 to 0.0015

**[MiniTamilBPETokenizer](https://github.com/nirmalpratheep/MiniTamilBPETokenizer)**
- Developed **Tamil BPE tokenizer** (5k vocab) achieving **3.1 char/token compression** ratio
- Curated corpus from **Project Madurai** and **Tamil Wikipedia**; deployed interactive demo via **Gradio**

### Agentic AI Projects
**[codingAgent](https://github.com/nirmalpratheep/codingAgent)**
- Built **8-Stage Bug Fixing Pipeline** using **Gemini 2.0 Flash** and **AST-based Dependency Analysis**
- Achieved **90%+ success rate** on tests by solving hidden dependencies and optimizing **token budget**

**[SWE-AgentBench](https://github.com/nirmalpratheep/SWE-AgentBench)**
- Developed **multi-agent benchmark system** with **3-agent architecture** (Evaluation Agent on Port 9001, Coding Agent on Port 9002, Reporting Agent on Port 9003) using agent-to-agent communication
- Integrated **Gemini 2.0 Flash LLM** for code generation and evaluation
- Built **Docker-based test isolation framework** for secure and reproducible test execution on **SWE-bench dataset**
- Implemented **automated repository setup** with Git clone, checkout, and patch operations for test case preparation
- Built **test execution engine** and **reporting system** for automated LLM performance benchmarking

### ML Projects
**[ImageNetClassifier](https://github.com/nirmalpratheep/ImageNetClassifier)**
- Trained **ResNet-50** on **ImageNet-1K (1.28M images)** achieving **77.4% Top-1 Accuracy**
- Optimized training with **CutMix**, **MixUp**, and **Random Erasing** augmentations; used **LR Finder** for optimal convergence

**[MNISTImageClassifier-ArchitectureExploration](https://github.com/nirmalpratheep/MNISTImageClassifier-ArchitectureExploration)**
- Conducted **CNN architecture search** achieving **99.50% accuracy** with only **17.3k parameters**
- Analyzed **Receptive Field vs Accuracy** trade-offs; optimized depth and **BatchNorm** for efficiency

**[CIFAR100-Resnet34](https://github.com/nirmalpratheep/CIFAR100-Resnet34)**
- Trained **ResNet-34** on **CIFAR-100**; achieved **76.7% accuracy**, **77.1% precision**
- Deployed interactive demo on **[HuggingFace Spaces](https://huggingface.co/spaces/nirmalpratheep/CIFAR100_ImageClassifier)**

## 🛠️ Skills

- **ML/RL:** PyTorch, HuggingFace, Stable Baselines 3, RL4CO, Gym, Ray
- **Programming:** Python, C++, Golang
- **Infrastructure:** Kubernetes, Docker, PostgreSQL, MongoDB
- **AI/LLM:** LLM Pre/PostTraining, Agentic Frameworks, GNN, Deep RL

## 🎓 Education

- **M.Eng** Electrical Engineering — U. of Cincinnati, 2012
- **B.Eng** Electronics & Comm. — Anna University, 2007

### 🏆 Honors
- **Top 15** — Innovate India Design Contest (ALTERA), 2007
- **Elite Mentorship Program** — AMD

## 📜 Certifications
Agentic Framework (HuggingFace) • Generative AI with LLMs (DeepLearning.AI) • ML OPS (DeepLearning.AI) • Machine Learning (Stanford) • Analytics Edge (MITx) • Parallel & Distributed Computing (Rice) • Kubernetes (Udacity) • BigData with Spark (Berkeley)
