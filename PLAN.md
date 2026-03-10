# Medical LLM Compression Plan (Pruning + Distillation + Quantization)

## Objective
Compress the earlier medical LLM checkpoint while preserving clinical-task quality and safety behavior.

## Success Criteria (set before experiments)
- Primary medical benchmark drop <= 1–2%
- No significant increase in hallucination / unsafe advice
- Meets deployment latency and VRAM targets

---

## 1) Pruning Plan

### Chosen Method
**Wanda-style activation-aware 2:4 semi-structured pruning**

### Why this method
1. **LLM-friendly importance scoring** (weight + activation signal) preserves critical pathways better than simple magnitude pruning.
2. **Practical speedups** are more likely with **2:4 semi-structured sparsity** on supported NVIDIA stacks.
3. **Good quality-efficiency tradeoff** before distillation/quantization.

### Scope and Order
- Prune first: **FFN/MLP linear layers** (highest redundancy)
- Keep dense initially: **embeddings, lm_head, layer norms**
- Prune attention projections conservatively after validation

### Sparsity Schedule
- Stage A: ~20%
- Stage B: ~35–40%
- Stage C (optional): ~50% only if eval gates pass

After each stage: short recovery tuning on medical data + gate evaluation.

---

## 2) Distillation Plan

### Teacher/Student
- **Teacher**: original full medical checkpoint
- **Student**: pruned checkpoint (or smaller dense target if needed)

### Loss
\[
\mathcal{L} = \alpha \cdot \text{CE} + \beta \cdot \text{KL}(z_s/T, z_t/T)
\]
Recommended start: `alpha=0.6`, `beta=0.4`, `T=2~4`

### Data Mix
- 60–70% medical instruction/QA
- 20–30% general text for language stability
- 10% safety/refusal examples for high-risk medical prompts

Goal: recover accuracy/calibration after pruning and preserve safe behavior.

---

## 3) Quantization Plan

### Chosen Method
**AWQ 4-bit (W4A16)**

### Why AWQ
- Activation-aware protection often preserves quality better than naive PTQ
- Strong memory reduction with good practical inference quality

### Policy
- Quantize most linear layers to 4-bit
- Keep sensitive modules at FP16/BF16:
  - embeddings
  - lm_head
  - layer norms
- Keep an INT8 baseline for safety-critical fallback comparison

---

## 4) End-to-End Execution Order
1. Baseline eval + safety eval
2. Pruning (Wanda 2:4) + short recovery tune
3. Distillation (teacher-guided)
4. Quantization (AWQ)
5. Final validation + deployment smoke tests

---

## 5) Evaluation Gates (every stage)
- Medical task metrics (EM/F1/Accuracy, task-dependent)
- Hallucination/error rate on clinical prompts
- Safety refusal correctness
- Calibration (e.g., ECE)
- Latency / throughput / VRAM

Rollback to previous checkpoint if thresholds are violated.

---

## 6) 10–14 Day Timeline
- **Day 1–2**: Baseline + eval harness freeze
- **Day 3–5**: Pruning sweeps (20/35/50%) + recovery
- **Day 6–9**: Distillation sweeps (alpha/beta/T/data mix)
- **Day 10–11**: AWQ variants + sensitivity exemptions
- **Day 12–14**: Safety red-team + final model selection

---

## Suggested Artifact Naming
- `ckpt_base`
- `ckpt_prune20`, `ckpt_prune35`, `ckpt_prune50`
- `ckpt_prune35_distill_v1`
- `ckpt_prune35_distill_awq4`

Include a `metrics.json` for each artifact with quality/safety/latency metadata.
