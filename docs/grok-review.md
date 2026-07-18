# Grok fact-check of the walkthrough pages (2026-07-18)

The six pages were audited by xAI's Grok (grok-4.5, headless) against the repo's report
files and scripts. Verbatim review below; resolution of each finding first.

## Resolution log

| # | Finding | Resolution |
|---|---------|-----------|
| 1 | Overview blurb said distillation climbs "from 61" (bf16 prune score) not 62 (measured distill start) | **Fixed** — blurb now says 62 → 80 |
| 2 | "800 training samples" unsupported by files in the audit workspace | **Kept** — verified against the sibling repo: `medical-llm/data/processed/medical_train.json` has exactly 800 records (Grok couldn't see that file) |
| 3 | KL direction reversed: `F.kl_div(log_softmax(student), softmax(teacher))` computes KL(teacher ‖ student) | **Fixed** — Chapter 3's loss block now reads KL(teacher/T ‖ student/T) |
| 4 | "~20-point dip and rebound" overstates the 13-point dip / 18-point rebound | **Fixed** — note now uses the exact figures |
| 5 | On-bar label "78%" vs table "77.8%" | **Fixed** — label now 77.8% |
| 6 | Success criterion quietly reworded from the plan's "1–2%" to "1–2 points" | **Fixed** — Chapters 1 and 5 now quote the plan's unit (equivalent on a 100-question exam, and now say so) |
| 7 | "PubMedQA" not named in this repo's files | **Kept** — the test set is prepared from PubMedQA per the sibling walkthrough's documented data prep |
| 8 | "sparsity preserved bit-for-bit" overclaims what the script guarantees | **Fixed** — now "training cannot un-prune them — the 2:4 sparsity survives intact" |
| 9 | Chapter 1 table "13.5" vs computed 13.49 | **Fixed** — table now 13.49 |
| 10 | INT8 bar label "−48%" vs table "−47.9%" | **Fixed** — bar now −47.9% |

---

# Review: Compression, visualized (hostile fact-check)

**Verdict.** The six HTML pages track the real prune → distill → quantize run closely: headline accuracies (74→77→68→61; distill 72/67/62→78/74/80; quant 79/81/81; AWQ 80), sparsity (38.92%), nonzero param counts, effective-memory GiB derived from nonzero × 2 / 2³⁰, bitsandbytes footprints (13.514 / 7.040 / 3.790 GB), KD hyperparameters (α=0.6, β=0.4, T=2, 80 steps, lr 1e-4, LoRA r=16 on attention), Wanda 2:4 on MLP only, eval prompt/grading, and the dense-storage / small-n caveats are all grounded in `sources/`. The main failures are a wrong distillation start score on the overview, an unsupported “800 training samples” claim repeated as fact, a KL argument order that does not match the script, and a few honesty/rounding nits. No broken internal links or chart-to-table data mismatches among the six pages.

---

## Findings

1. **INACCURACY** — `pages/index.html` (Chapter 3 blurb)  
   **Quoted text:** “Accuracy climbs from 61 back to 80 in 80 steps.”  
   **Source:** `sources/reports/iterative_distillation_metrics.json` (stage3: before `correct` 62, after 80); `sources/reports/iterative_pruning_metrics.json` (stage3 eval 61 is the separate bf16 prune eval).  
   **Correct:** Distillation’s measured climb for stage 3 is **62 → 80** under the NF4 student setup. 61 is the bf16 pruning-stage score, not the distillation “before” point. Chapter 3’s lede (“climbed from 62 to 80/100”) and Chapter 5’s provenance table get this right; the overview blurb does not.

2. **INACCURACY** — `pages/03-distillation.html` (body, SVG label, code comment)  
   **Quoted text:** “only 800 examples and 80 optimizer steps”; SVG “800 medical Q&A”; “same 800 medical training samples the original finetune used”.  
   **Source:** `sources/run_iterative_distillation.py` loads the full `medical_train.json` via `train_data = load_json(TRAIN_JSON)` with **no length recorded** in any report, JSON metric file, `FINAL_SUMMARY.md`, `PLAN.md`, or `README.md`. Nothing in `sources/` states 800.  
   **Correct:** Dataset size is **unsupported** in the provided ground truth. The script uses whatever length `medical_train.json` has, for **80 steps** at batch size 1 per stage (not “800 examples” as a documented fact). Drop the number or cite an actual train-set cardinality from the data artifact.

3. **INACCURACY** — `pages/03-distillation.html` (loss diagram + code block)  
   **Quoted text:** `loss = 0.6·CE + 0.4·KL·T²` with `KL(student/T ‖ teacher/T)`.  
   **Source:** `sources/run_iterative_distillation.py` lines 163–180:
   ```python
   kl_token = F.kl_div(
       F.log_softmax(s_shift, dim=-1),
       F.softmax(t_shift, dim=-1),
       reduction='none',
   ).sum(dim=-1)
   ...
   kl = kl * (TEMP * TEMP)
   loss = ALPHA * ce + BETA * kl
   ```
   PyTorch `kl_div(input, target)` computes **KL(target ‖ input)**. With `input = log_softmax(student)` and `target = softmax(teacher)`, that is **KL(teacher ‖ student)**, not KL(student ‖ teacher).  
   **Correct:** Write **KL(teacher/T ‖ student/T)** (or “soft CE of student to teacher targets”) × T², scaled by 0.4, plus 0.6 × hard CE. Weights α/β/T and the T² factor themselves match the script.

4. **INACCURACY** (honesty / framing) — `pages/index.html`  
   **Quoted text:** “the ~20-point dip and rebound are the signal.”  
   **Source:** `sources/reports/iterative_pruning_metrics.json` (baseline 74 → stage3 61 = **−13**); `sources/reports/iterative_distillation_metrics.json` (stage3 +0.18 = **+18** from 62→80).  
   **Correct:** The prune **dip vs baseline is 13 points** (16 from the stage-1 peak of 77). The distill **rebound is ~18 points**. Calling the dip “~20-point” overstates the drop; “~13-point dip and ~18-point rebound” matches the reports. Chapter 1’s “13-point pruning drop and 18-point distillation rebound” is the accurate phrasing.

5. **NITPICK** — `pages/02-pruning.html` (stacked-bar segment label vs table/prose)  
   **Quoted text:** chart label `MLP · 5.64B (78%)` vs table/prose **77.8%** / “nearly 78%”.  
   **Source:** `sources/reports/pruning_step_metrics.json` `target_layers_considered_params` = 5,637,144,576 of `total_params` 7,241,732,096 → **77.84%**.  
   **Correct:** Use **77.8%** (or “~78%”) consistently in the on-bar label; 78% without a tilde is a rounding stretch next to a table that shows 77.8%.

6. **NITPICK** — `pages/01-baseline.html` / `pages/05-results.html` vs plan wording  
   **Quoted text:** “accuracy drop within 1–2 points of baseline” / “Accuracy within 1–2 points of baseline”.  
   **Source:** `sources/PLAN.md`: “Primary medical benchmark drop <= **1–2%**”.  
   **Correct:** Prefer the plan’s **1–2%** wording, or explicitly equate “1–2 percentage points on the 100-example accuracy” if that is the intended reading. As written, the pages quietly rewrite the pre-registered criterion’s unit.

7. **NITPICK** — `pages/01-baseline.html` (and later “PubMedQA” mentions)  
   **Quoted text:** “the same 100 held-out PubMedQA questions”.  
   **Source:** Scripts and reports only name `medical_test.json` / a 100-example yes/no/maybe eval (`run_iterative_pruning.py` `TEST_JSON`, all metrics `total: 100`). **No source file names PubMedQA.**  
   **Correct:** “100 held-out medical test examples (`medical_test.json`, yes/no/maybe)” unless a source in-repo identifies the dataset as PubMedQA.

8. **NITPICK** — `pages/03-distillation.html` (callout)  
   **Quoted text:** “the 2:4 sparsity is preserved **bit-for-bit**.”  
   **Source:** `sources/run_iterative_distillation.py` freezes the NF4 base and attaches LoRA only to `q_proj/k_proj/v_proj/o_proj` (MLP not in `target_modules`), so MLP weights get **no optimizer updates**. It does **not** re-assert or re-save an exact 2:4 mask after NF4 load.  
   **Correct:** “MLP base weights are frozen (no LoRA on up/gate/down), so training cannot un-prune them.” “Bit-for-bit” overclaims what the script guarantees under 4-bit packing.

9. **NITPICK** — `pages/01-baseline.html` (VRAM table vs chart data)  
   **Quoted text:** table cell “This model, fp16 weights” **13.5**; chart `value: 13.49`, label “13.5 GB”.  
   **Source / recompute:** nonzero 7,241,732,092 × 2 / 2³⁰ = **13.4888… → 13.49** GiB (`FINAL_SUMMARY.md` also uses ~13.49 GiB).  
   **Correct:** Table should show **13.49** (or both prose and table stick to one rounded form and note the GiB basis).

10. **NITPICK** — `pages/04-quantization.html` (INT8 bar label vs table)  
    **Quoted text:** bar label `7.04 GB · −48%` vs table `−47.9%`.  
    **Source:** `sources/reports/quantization_step_metrics.json` `memory_saving_vs_fp16` = 0.479097… → **47.91%**.  
    **Correct:** **−47.9%** (or −48% with a tilde) on both the bar and the table.

---

## Categories with no findings

- **ERROR (hard numeric contradiction of report figures):** None. Pipeline accuracies, sparsities, nonzero counts, quant memory/accuracy, and AWQ 80 vs NF4 81 match `sources/reports/*`.
- **Mechanics:** No broken links among the six pages; chapter titles match content; chart element ids (`chart-teaser`, `chart-vram`, `chart-params`, `chart-acc`, `chart-mem`, `chart-distill`, `chart-journey`) all have matching `div`s; chart series data match their sibling “View as table” tables where both exist. (Missing `assets/*` is out of scope per audit rules.)
