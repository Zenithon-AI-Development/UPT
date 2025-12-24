# Model Benchmarking Results

This document contains benchmarking results for various model architectures trained on The WELL dataset for next-timestep prediction tasks.

## Methodology

### Training Configuration

- **Task**: Next timestep prediction from previous 4 timesteps
- **Training Metric**: Relative L2 error
- **Dataset**: The WELL / Zen WELL Garden (using same presplits as Polymathic benchmarks)
- **Data Splits**: Train/validation/test presplits from The WELL dataset
- **Hardware**: Each model trained on 1 GPU (Nvidia L4) under full utilization unless specified otherwise
- **Normalization**: Each channel normalized separately to 0 mean and standard deviation 1
- **Hyperparameters**: Unless specified otherwise identical to the configuration from UPT/src/yamls/sim10k/upt/e100_lr5e5_17M_lat512_fp32.yaml

### Evaluation Metrics

- **Relative L2 Error**: Final relative L2 error on test split
- **Relative L1 Error**: Final relative L1 error on test split
- **Time per Epoch**: Wall-clock time per training epoch
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

### turbulent_radiative_layer_2D (https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/)

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch (s) | Time to 50% Val Error (s) | Time to 20% Val Error (s) | Time to 10% Val Error (s) | Time to 5% Val Error (s) | # Parameters (Cond/Enc/Proc/Dec) | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|--------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------| 
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | 25.4% | 15.5% | 944.69s | 0.96hrs | - | - | - | 17.189.092 (1.477.632, 4.054.272, 5.356.992, 6.300.196) | 25.26ms | 2.58ms | 88.37ms |

### helmholtz_staircase (https://polymathic-ai.org/the_well/datasets/helmholtz_staircase/)

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch (s) | Time to 50% Val Error (s) | Time to 20% Val Error (s) | Time to 10% Val Error (s) | Time to 5% Val Error (s) | # Parameters (Cond/Enc/Proc/Dec) | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|--------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------|
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | | | 22774.9s | 45332.0s | 158391.0s | 226381.0s | 431227.0s | 17.189.092 (1.477.632, 4.054.272, 5.356.992, 6.300.196) |25176.74 ms |  4.57 ms | 149.43 ms |
| quadtree UPT | QuadtreeTransformerPerceiver | TransformerModel | TransformerPerceiver | | | 7013.4s | 5400.0s | 26607.0s | 78623.0s | - | 14.887.202 (1.477.632, 1.752.576, 5.356.992, 6.300.002) | 6.77ms | 3.25ms | 80.84ms |

### Zpinch (multiple resolutions --> only 1024  =32^2 supernodes equal to lowest resolution total node count, https://zenithon-zengarden-datasets.web.app/flash_zpinch_2d_no_heating/)

| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch (s) | Time to 50% Val Error (s) | Time to 20% Val Error (s) | Time to 10% Val Error (s) | Time to 5% Val Error (s) | # Parameters (Cond/Enc/Proc/Dec) | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|---------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------|
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | 28.3% | 13.9% | 2146.2s | 0 | - | - | - | 8.845.927 (738.816, 2.859.936, 2.697.024, 2.550.151) | 5.37ms | 1.80ms | 6.35ms |
| quadtree UPT | QuadtreeTransformerPerceiver | TransformerModel | TransformerPerceiver | | | 2778.0s | 2872.0s | 6208.0s | - | - | 17.107.110 (1.477.632, 3.972.096, 5.356.992, 6.300.390) | 10.59ms | 4.21ms | 8.24ms |

### MHD (https://polymathic-ai.org/the_well/datasets/MHD_256/)
| Model | Encoder | Processor | Decoder | Rel L2 (test) | Rel L1 (test) | Time/Epoch (s) | Time to 50% Val Error (s) | Time to 20% Val Error (s) | Time to 10% Val Error (s) | Time to 5% Val Error (s) | # Parameters (Cond/Enc/Proc/Dec) | Inference - Encoder (ms) | Inference - Processor (ms) | Inference - Decoder (ms) |
|-------|---------|-----------|---------|---------------|---------------|---------------------------|----------------------|----------------------|---------------------|---------------------|---------------|---------------------------|----------------------------|--------------------------|
| standard UPT | PoolTransformerPerceiver | TransformerModel | TransformerPerceiver | | | | | | | | | | | |
| quadtree UPT | QuadtreeTransformerPerceiver | TransformerModel | TransformerPerceiver | | | | | | | | | | | |
