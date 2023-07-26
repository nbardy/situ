# situ

New Model Architecture for Upscaling with Modern Transformers

# Plan

Step 1. Take the approach of modeling discrete tokens in the VQ-GAN domain of maskGIT and apply it to upscaling

- [ ] Write JAX training Code for finetuning existing maskGIT
- [ ] Train Baseline
      - [ ] Finetune the transformer on the upscaling task.
- [ ] Add cross conditioning with text
- [ ] Add flash attention with jax


Step 2.

- [ ] Add local blocks Depthwise Separable Convultions and Multihead-Attention) + global residual(cross attention with dilated residual + text)

Step 3.
- [ ] Speed up with new runtime algo.
