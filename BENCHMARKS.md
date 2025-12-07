# Model Benchmarking Results

This document contains benchmarking results for various model architectures trained on The WELL dataset for next-timestep prediction tasks.

## Methodology

### Training Configuration

- **Task**: Next timestep prediction from previous 4 timesteps
- **Training Metric**: Relative L2 error
- **Dataset**: The WELL / Zen WELL Garden (using same presplits as Polymathic benchmarks)
- **Data Splits**: Train/validation/test presplits from The WELL dataset
- **Hardware**: Each model trained on 1 GPU under full utilization
- **Normalization**: Each channel normalized separately to 0 mean and standard deviation 1

### Evaluation Metrics

- **Relative L2 Error**: Final relative L2 error on test split
- **Relative L1 Error**: Final relative L1 error on test split
- **Time per Epoch**: Wall-clock time per training epoch, broken down by component:
  - Encoder time
  - Processor (latent core) time
  - Decoder time
- **Convergence Time**: Time to reach specific relative error thresholds:
  - 50% relative error
  - 20% relative error
  - 10% relative error
  - 5% relative error
- **Model Size**: Total number of trainable parameters
- **Inference Time**: Per-sample inference time, broken down by component:
  - Encoder inference time
  - Processor inference time
  - Decoder inference time

### Notes

- All timing measurements are in seconds unless otherwise specified
- Convergence times are measured from training start to first epoch where the threshold is reached
- Time per epoch is averaged over all training epochs
- Parameters count includes all trainable parameters in the model

## Benchmark Results

### turbulent_radiative_layer_2D

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch - Encoder (s) | Time/Epoch - Processor (s) | Time/Epoch - Decoder (s) | Time to 50% Error (s) | Time to 20% Error (s) | Time to 10% Error (s) | Time to 5% Error (s) | # Parameters | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|---------------------------|----------------------------|--------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------|
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | | | | | | | | | | | | | |

### helmholtz_staircase

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch - Encoder (s) | Time/Epoch - Processor (s) | Time/Epoch - Decoder (s) | Time to 50% Error (s) | Time to 20% Error (s) | Time to 10% Error (s) | Time to 5% Error (s) | # Parameters | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|---------------------------|----------------------------|--------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------|
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | | | | | | | | | | | | | |

### Zpinch (multiple resolutions on uniform grids)

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch - Encoder (s) | Time/Epoch - Processor (s) | Time/Epoch - Decoder (s) | Time to 50% Error (s) | Time to 20% Error (s) | Time to 10% Error (s) | Time to 5% Error (s) | # Parameters (Cond/Enc/Proc/Dec) | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|---------------------------|----------------------------|--------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------|
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | 29.7 % | 15.4% |  |  |  |  | - | - | - | 8,845,927 (738816,
(2859936, 2697024, 2550151) | | | |

