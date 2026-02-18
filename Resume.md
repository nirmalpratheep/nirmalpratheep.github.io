# Nirmal Pratheep Natarajan
**AI/ML Research Engineer**
[nirmalpratheep@gmail.com](mailto:nirmalpratheep@gmail.com) | [GitHub](https://github.com/nirmalpratheep) | [LinkedIn](https://linkedin.com/in/nirmalpratheep)

---

AI/ML Research Engineer with 13+ years at AMD & Xilinx, specializing in **implementing and optimizing ML research** from paper to production. Hands-on experience with **LLM pre-training & alignment** (SFT, GRPO, RLHF), **deep reinforcement learning** for combinatorial optimization, and **GPU kernel-level performance optimization** (Triton, Flash Attention, Nsight profiling). Proven track record of translating research ideas into working systems, designing experiments, and publishing results at peer-reviewed conferences.

## Research & Publications

- **Deep RL for FloorPlan Optimization** — GTAC'25 & SPS Tech Conference *(Finalist, arXiv pending)*
  - Formulated FPGA floorplan directive optimization as an RL problem; GIN feature extraction on 15M-node netlists; 2% placement QoR gain over manual expert tuning
- **ML-based Delay Prediction** — GTAC'22 AMD Tech Conference *(Finalist)*
  - ML delay prediction models and GNN-based design complexity analysis with automated fine-tuning, model monitoring, and drift detection
- **LLM Alignment & Reasoning via RL** — [Code](https://github.com/nirmalpratheep/Alignment_and_Reasoning_RL) | [W&B](https://wandb.ai/nirmalpratheep-self/math-grpo-trl)
  - End-to-end alignment pipeline (Baseline → SFT → GRPO RL) on Qwen 2.5 Math 1.5B; 14.2× zero-shot accuracy gain with systematic ablation
- **Adaptive OFDM Pilots** — IEEE WAMICON 2009

## Benchmarks & Key Results

| Area | Metric | Detail |
|------|--------|--------|
| LLM Alignment | **14.2x** reasoning gain | Qwen 2.5 Math 1.5B: 2.84% → 40.46% zero-shot accuracy (Baseline → SFT → GRPO) |
| Pre-training | **33%** throughput gain | 1B parameter model: custom Triton kernels, fused Flash Attention, Nsight profiling |
| Deep RL | **2%** placement QoR | RL directive optimization on 15M-node graphs, replacing manual expert tuning |
| Agentic AI | **90%+** success rate | Multi-stage agentic pipeline with graph-based orchestration, AST analysis, self-correction |
| Systems | **3x** throughput | Client/server architecture for simulation on 10nm/7nm/2nm FPGA nodes |
| Systems | **20x** scale | Divide-and-conquer parallel processing via LSF Farm |

## Professional Experience

### AMD (formerly Xilinx) — Senior Staff / Research Engineer
**San Jose, CA | 2012 – Present**

#### Research & ML Systems
- Researched and implemented **Deep RL for directive optimization**: designed environment, reward shaping, and GIN-based feature extraction on netlists up to **15M nodes**; published at **GTAC'25 (Finalist)**
- Built **Ray-based distributed training** infrastructure with Grid, ASHA, and PBT hyperparameter search for systematic experiment management and scalable RL training
- Developed **ML delay prediction algorithms** and **GNN-based design complexity models** with automated fine-tuning, model monitoring, and drift detection; published at **GTAC'22 (Finalist)**
- Built **Agentic AI framework** with multi-step orchestration and LLMs for **autonomous triage** with iterative self-correction and Dockerized evaluation

#### Performance Engineering & Infrastructure
- **Led 10+ engineer team** across global sites for simulation delay capture tooling on **10nm/7nm/2nm FPGA nodes**
- Architected **client/server system** (Boost Asio + Google Protobuf) enabling concurrent multi-capture with **3x throughput** improvement
- Designed **divide-and-conquer parallel processing** system via LSF Farm, enabling **20x larger chip versions**
- Built **graph compression pipeline** processing 3.5B datapoints, reducing 1B instance paths to 500K patterns with Python analytics and visualization
- Developed tool profiling, linters, debugging dashboards, and **YAML semantic verifier** auto code generation for HW/SW validation

## Research Engineering Projects

### LLM & Pre-training
**[LLM Alignment & Reasoning RL](https://github.com/nirmalpratheep/Alignment_and_Reasoning_RL)**
- Implemented complete alignment pipeline for Qwen 2.5 Math 1.5B: Baseline → SFT → GRPO RL achieving 14.2× accuracy gain (2.84% → 40.46%)
- TRL GRPOTrainer with vLLM colocate mode; dual-GPU Optuna + ASHA hyperparameter optimization
- W&B Experiments: [HPO](https://wandb.ai/nirmalpratheep-self/math-sft-optuna-asha) | [SFT](https://wandb.ai/nirmalpratheep-self/math-sft) | [GRPO](https://wandb.ai/nirmalpratheep-self/math-grpo-trl)

**1B Seed Model Pre-training**
- Optimized 1B Dense parameter model (75% GDN, 25% GSA with midpoint reversibility). 33% throughput gain via custom Triton kernels, fused Flash Attention. Profiled with Nsight Systems/Compute.

**[SmolLM v2 Pre-training](https://github.com/nirmalpratheep/SmolLMv2-PreTrain)**
- 135M parameter model on FineWeb-Edu with Flash Attention. ~40k tokens/sec (BF16); loss 11.6 → 0.0015

**[MiniTamilBPETokenizer](https://github.com/nirmalpratheep/MiniTamilBPETokenizer)**
- Tamil BPE tokenizer (5k vocab) achieving 3.1 char/token compression ratio

### Agentic AI & Evaluation
**[Agentic Coding Pipeline](https://github.com/nirmalpratheep/codingAgent)**
- 8-stage bug fixing pipeline with graph-based orchestration, AST-based dependency analysis, iterative self-correction. 90%+ success rate.

**[SWE-Agent Benchmark](https://github.com/nirmalpratheep/SWE-AgentBench)**
- 3-agent evaluation architecture with Docker-based test isolation on SWE-bench for reproducible LLM benchmarking

### Deep Learning
**[ImageNet Classifier](https://github.com/nirmalpratheep/ImageNetClassifier)**
- ResNet-50 on ImageNet-1K achieving 77.4% Top-1 accuracy. CutMix, MixUp, Random Erasing + LR Finder.

**[RL Car Navigation](https://github.com/nirmalpratheep/RL-CarNavigationAgent)** | **[CIFAR-100 ResNet-34](https://github.com/nirmalpratheep/CIFAR100-Resnet34)** | **[MNIST Architecture Search](https://github.com/nirmalpratheep/MNISTImageClassifier-ArchitectureExploration)**

## Skills

- **ML Frameworks:** PyTorch, HuggingFace, TRL, vLLM, DeepSpeed, FSDP
- **RL & Agents:** Stable Baselines 3, Ray, Gym, multi-agent orchestration
- **GPU & Performance:** Triton, Flash Attention, Nsight Systems/Compute, CUDA, mixed-precision
- **Languages:** Python, C++, Golang
- **Infrastructure:** Kubernetes, Docker, Ray, LSF, W&B, Optuna

## Education

- **M.Eng** Electrical Engineering — University of Cincinnati, 2012
- **B.Eng** Electronics & Communication — Anna University, 2007

### Honors
- **Top 15** — Innovate India Design Contest (ALTERA), 2007
- **Elite Mentorship Program** — AMD

## Certifications
Triton Kernel Dev on AMD Instinct GPUs • LLM Serving with vLLM & MI300X • Agentic Framework (HuggingFace) • Generative AI with LLMs (DeepLearning.AI) • ML Ops (DeepLearning.AI) • Machine Learning (Stanford) • Analytics Edge (MITx) • Parallel & Distributed Computing (Rice) • Kubernetes (Udacity) • Big Data with Spark (Berkeley)
