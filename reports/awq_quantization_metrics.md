# AWQ Quantization Step Report

- Source model: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/iterative-wanda-2of4-distilled-merged/stage3_add_down_proj`
- AWQ output: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/awq-w4a16-stage3`
- Recipe: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/awq_recipe_w4a16.yaml`

## Results
### fp16
- Accuracy: 79/100 = 0.7900
- Memory footprint: 13.489 GB

### awq_w4a16
- Accuracy: 80/100 = 0.8000
- Accuracy delta vs fp16: +0.0100
- Memory footprint: 16.891 GB
- Memory saving vs fp16: -25.22%