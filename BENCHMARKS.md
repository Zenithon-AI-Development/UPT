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
- **Time per Epoch**: Total wall-clock time per training epoch measured during training
- **Convergence Time**: Time from training start to first epoch where validation relative error threshold is reached:
  - 50% relative error
  - 20% relative error
  - 10% relative error
  - 5% relative error
- **Model Size**: Total number of trainable parameters, broken down by component (Conditioner/Encoder/Processor/Decoder)
- **Inference Time**: Per-sample inference time, broken down by component:
  - Encoder inference time
  - Processor inference time
  - Decoder inference time

### Notes

- All timing measurements are in seconds unless otherwise specified
- Convergence times are measured on validation set from training start to first epoch where the threshold is reached
- Time per epoch is averaged over all training epochs
- Parameters count includes all trainable parameters in the model

## Benchmark Results

### turbulent_radiative_layer_2D

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch (s) | Time to 50% Val Error (s) | Time to 20% Val Error (s) | Time to 10% Val Error (s) | Time to 5% Val Error (s) | # Parameters (Cond/Enc/Proc/Dec) | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|--------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------| 
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | 25.4% | 15.5% | 944.69s | 0.96hrs | - | - | - | 17.189.092 (1.477.632, 4.054.272, 5.356.992, 6.300.196) | 25.26ms | 2.58ms | 88.37ms |

### helmholtz_staircase

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch (s) | Time to 50% Val Error (s) | Time to 20% Val Error (s) | Time to 10% Val Error (s) | Time to 5% Val Error (s) | # Parameters (Cond/Enc/Proc/Dec) | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|--------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------|
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | | | | | | | | | 17.189.092 (1.477.632, 4.054.272, 5.356.992, 6.300.196) | | | |

### Zpinch (multiple resolutions on uniform grids)

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch (s) | Time to 50% Val Error (s) | Time to 20% Val Error (s) | Time to 10% Val Error (s) | Time to 5% Val Error (s) | # Parameters (Cond/Enc/Proc/Dec) | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|---------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------|
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | 28.3% | 13.9% | 2146.2s | 0 | - | - | - | 8.845.927 (738.816, 2.859.936, 2.697.024, 2.550.151) | 5.37ms | 1.80ms | 6.35ms |

