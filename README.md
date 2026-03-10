# fedvlm-lora-pathvqa

Federated LoRA fine-tuning for a Vision-Language Model (VLM) on **PathVQA** (medical pathology VQA).

## Project Goal
This repository aims to:
- Fine-tune a VLM (e.g., Qwen VLM) on PathVQA using **LoRA** (parameter-efficient fine-tuning).
- Extend the training to **federated learning** settings, focusing on:
  - **Heterogeneous LoRA rank** (different clients use different ranks)
  - **Aggregation** for heterogeneous LoRA updates
  - (Optional) multimodal FL baselines and ablations

## Status (MVP)
- [x] Single-node LoRA fine-tuning pipeline on PathVQA (Unsloth + TRL SFTTrainer)
- [ ] Federated simulation (client split, local training, server aggregation)
- [ ] Heterogeneous rank experiments (rank sweep + aggregation)
- [ ] Evaluation (beyond eval_loss: EM/accuracy for VQA)

---

## Environment

### Hardware
- **GPU is required** for Unsloth.
- Recommended: NVIDIA GPU (A100/V100/T4). CPU-only runtime will fail.

### Dependencies
Core packages:
- `torch` (CUDA build)
- `unsloth`, `trl`, `datasets`
- `transformers`, `peft`, `accelerate`
- `bitsandbytes` (for `adamw_8bit` optimizer)

Install (example):
```bash
pip install unsloth trl datasets transformers peft accelerate bitsandbytes
