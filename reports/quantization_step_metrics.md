# Quantization Step Report

- Source model: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/iterative-wanda-2of4-distilled/stage3_add_down_proj`
- Note: Quantization done with bitsandbytes (INT8/NF4). AWQ package not installed in env.

## Results
### fp16
- Accuracy: 79/100 = 0.7900
- Accuracy delta vs fp16: +0.0000
- Memory footprint: 13.514 GB
- Memory saving vs fp16: 0.00%

### int8
- Accuracy: 81/100 = 0.8100
- Accuracy delta vs fp16: +0.0200
- Memory footprint: 7.040 GB
- Memory saving vs fp16: 47.91%

### nf4
- Accuracy: 81/100 = 0.8100
- Accuracy delta vs fp16: +0.0200
- Memory footprint: 3.790 GB
- Memory saving vs fp16: 71.96%

## Recommendation
- nf4