# 🩺 Llama-R1-MedRAG: Reinforcement Learning for Medical Reasoning using DeepSeek-R1-Distill-Llama-8B
- Due to privacy and protection of the project with other collaborators, the source code is hidden currently.

## 📘 Overview

**Llama-R1-MedRAG** is a research project exploring how **Reinforcement Learning (RL)** and **Retrieval-Augmented Generation (RAG)** can be combined to improve **medical reasoning and diagnostic accuracy** in **vision-language models (VLMs)**.

The model builds on **DeepSeek-R1-Distill-Llama-8B**, leveraging its strong reasoning ability as a foundation.  
We fine-tune it on **medical QA datasets** and design a **reward-driven optimization loop** to improve **faithfulness**, **clinical soundness**, and **reasoning depth** in medical contexts.

---

## 🎯 Motivation

While modern LLMs and VLMs demonstrate remarkable general reasoning abilities, they often **hallucinate** and **fail to reason reliably** over structured medical data such as EHR tables, radiology images, and clinical notes.  
This project aims to bridge that gap through **reinforcement learning and retrieval grounding**, focusing on **medical diagnosis reasoning**.

We aim to:
- Improve **diagnostic accuracy** through better long-form reasoning.
- Reduce **hallucination and unsafe recommendations**.
- Integrate **retrieval-based domain grounding** (e.g., MedRAG, RAGAS) with **RL-based optimization** for factual consistency.

---

## 🧠 Core Methodology

### 1. **Supervised Fine-Tuning (SFT)**
We begin by fine-tuning **DeepSeek-R1-Distill-Llama-8B** on medical QA datasets such as:
- [**PMC-VQA**](https://arxiv.org/abs/2305.10415) — 227K VQA samples from PubMed Central
- [**EHRXQA**](https://physionet.org/content/ehrxqa/1.0.0/) — multi-modal EHR QA pairs with Chest X-rays and SQL-based answers
- [**MedQA**](https://arxiv.org/abs/2009.13081) — multiple-choice medical exam dataset with reasoning evidence

This stage provides the model with foundational medical knowledge and exposure to diagnostic reasoning tasks.

### 2. **Retrieval-Augmented Generation (RAG)**
To reduce hallucination and improve factual grounding, we introduce a **self-ask retrieval module** that allows the model to query external RAG frameworks such as [**MedRAG**](https://aclanthology.org/2024.findings-acl.386/) or [**RAGAS**](https://github.com/explodinggradients/ragas).  
This step augments the model’s context with relevant medical literature, symptoms, and disease information before generating its reasoning.

### 3. **Reinforcement Learning (RL)**
Finally, we apply **Proximal Policy Optimization (PPO)** or **Generalized Reversed PPO (GRPO)** to fine-tune the model’s reasoning behavior.  
A hybrid **reward function** evaluates:
- **Medical correctness** (using domain models such as BioClinicalBERT or MedAlpaca)
- **Reasoning depth and coherence**
- **Safety and factual grounding**

---

## 🧩 System Design

### Training Pipeline
```
Medical QA Dataset (PMC-VQA, EHRXQA, MedQA)
        ↓
   Supervised Fine-Tuning (SFT)
        ↓
     Retrieval-Augmented Generation (RAG)
        ↓
 Reinforcement Learning (PPO/GRPO)
        ↓
   Policy Model (MedRAG-RL)
```

### Reward Model Integration
- **Local domain-tuned evaluators**: BioClinicalBERT, PubMedBERT, MedAlpaca  
- **External reasoning APIs**: DeepSeek-R1, GPT-4, Med-PaLM 2  
- **RAG support**: MedRAG, RAGAS for retrieval-based reward shaping

---

## 📊 Evaluation

We evaluate performance on real-world and exam-style benchmarks following prior works such as **Med-R1**, **MedVLM-R1**, and **Patho-AgenticRAG**.

**Metrics:**
- **BLEU** — lexical similarity to ground truth  
- **Accuracy** — correctness in multiple-choice QA  
- **Faithfulness & safety** — assessed via reward model scoring  
- **Qualitative reasoning analysis** — visualization of thought chains and intermediate reasoning

Performance will be compared to:
- **Baseline:** DeepSeek-R1-Distill-Llama-8B  
- **SFT-only model**  
- **RAG-only model**  
- **Full RL-augmented MedRAG**

---

## 🧪 Hypotheses

1. Incorporating **RAG** significantly improves domain grounding and factual accuracy, even without reinforcement learning.  
2. Combining **RAG + RL** leads to deeper, more reliable diagnostic reasoning through retrieval-enhanced policy optimization.  
3. Reinforcement learning encourages longer and more deliberate “thinking” (test-time compute), improving diagnosis accuracy.

---

## 🧩 Related Work

This project draws inspiration from:
- [**Med-R1**](https://arxiv.org/abs/2503.13939) — RL for generalizable medical reasoning  
- [**MedVLM-R1**](https://arxiv.org/abs/2503.15045) — RL for incentivizing visual-text reasoning  
- [**Patho-AgenticRAG**](https://arxiv.org/abs/2504.10640) — RL-guided multimodal RAG for pathology  
- [**Med-R3**](https://arxiv.org/abs/2505.09287) — Progressive RL for retrieval-augmented reasoning  

---

### 👨‍⚕️ Author
**Thomas He** — University of Michigan, ECE 598 (Fall 2025)
