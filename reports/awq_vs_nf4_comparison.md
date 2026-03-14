# AWQ vs NF4 Comparison (Distilled Stage3)

## Sources
- AWQ report: `reports/awq_quantization_metrics.json`
- NF4 report: `reports/quantization_step_metrics.json`

## Headline
- **AWQ W4A16:** 80/100 = **0.8000**
- **NF4 (bnb 4-bit):** 81/100 = **0.8100**
- **Delta (AWQ - NF4):** **-0.0100** (AWQ lower by 1 sample)

## Notes on memory numbers
- AWQ report shows `16.891 GB` and NF4 report shows `3.790 GB`, but these are from **different quant backends/loaders** and are **not directly comparable** using this `get_memory_footprint()` path.
- Treat the memory section as directional within each backend, not as strict cross-backend truth.

## Practical recommendation
- On current 100-sample eval, **NF4 remains the best default** (slightly better quality + known efficient bnb runtime path).
- Use AWQ only if you need AWQ/compressed-tensors ecosystem compatibility or specific serving constraints.

## Artifact paths
- AWQ model: `artifacts/awq-w4a16-stage3`
- NF4 benchmark source report: `reports/quantization_step_metrics.json`
