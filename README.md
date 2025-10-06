# 🩺 Med-RLHF: Reinforcement Learning for Medical Reasoning using LLaVA-7B
<img width="2404" height="1029" alt="image" src="https://github.com/user-attachments/assets/92a9e9a6-e752-4684-aa47-d6985574e934" />
- Overview of the project is shown below.
- Due to privacy and protection of the project with other collaborators, the source code is hidden currently.
## 📘 Overview

**Med-RLHF** is an experimental research project exploring how **Reinforcement Learning with Human Feedback (RLHF)** can be adapted for **medical reasoning tasks** using large vision-language models (VLMs).  
We use **LLaVA-7B** as the base model and design a reward-driven training loop that encourages the model to generate safe, coherent, and clinically sound medical responses.

This project aims to evaluate whether modern RLHF techniques—specifically **Proximal Policy Optimization (PPO)**—can improve the alignment of general-purpose models with **domain-specific medical knowledge**, without relying on manually curated reinforcement signals.

---

## 🎯 Motivation

Medical decision-making requires precision, reasoning, and factual consistency. However, most foundation models are optimized for general conversation, not clinical reliability.  
This project investigates whether we can **close that gap** by using reinforcement learning to align models with medical reward signals.

By combining **vision-language models** (for understanding medical imagery) and **domain-specific reward models**, we aim to:
- Enhance model **reasoning** over structured and unstructured medical data.
- Improve **faithfulness** to evidence-based medical knowledge.
- Reduce **hallucinations** and unsafe recommendations.

---

## 🧠 Core Idea

At a high level, Med-RLHF follows the same principles as OpenAI’s RLHF pipeline:
1. **Supervised Fine-Tuning (SFT)**: Fine-tune a base model (LLaVA-7B) on medical image–text data to learn foundational reasoning.
2. **Reward Modeling**: Train or integrate a *reward model* that scores responses based on medical correctness, safety, and coherence.
3. **Reinforcement Learning (PPO)**: Use the reward model’s feedback to iteratively optimize the policy model.

However, unlike conventional RLHF pipelines, **our reward model and value function may be accessed via APIs**, allowing hybrid evaluation using:
- Local **domain-tuned classifiers** (e.g., BioClinicalBERT, PubMedBERT, MedAlpaca)
- Remote **reasoning LLMs** (e.g., DeepSeek-R1, GPT-4, or Med-PaLM 2) on synthetic or de-identified prompts

---

## 🧩 System Design

### Training Flow
```mermaid
flowchart TD
    A[Medical Prompt / Image] --> B[LLaVA-7B Policy Model]
    B --> C[Generated Response]
    C --> D[Reward Model (BioBERT / MedReward / API Evaluator)]
    D --> E[Reward Signal]
    E --> F[PPO Optimizer]
    F --> G[Policy Update]
    G --> B
