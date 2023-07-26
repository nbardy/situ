# situ

New Model Architecture for Upscaling with Modern Transformers

# Plan

Step 1. Take the approach of modeling discrete tokens in the VQ-GAN domain of maskGIT and apply it to upscaling

- [ ] Write JAX training Code for finetuning existing maskGIT
- [ ] Train Baseline - [ ] Finetune the transformer on the upscaling task.
- [ ] Add cross conditioning with text
- [ ] Add flash attention with jax

Step 2. Make it better

- [ ] Add flash attention with jax
- [ ] Append layers to the end of the VQ-GAN decoder and train in series (Done in the muse paper)

Step 3. Add some ideas from transformer papers for high resolution upscaling.

- [ ] Add local blocks Depthwise Separable Convultions and Multihead-Attention) + global residual(cross attention with dilated residual + text)

Step 4

- [ ] Speed up with new runtime algo.

## Attributes

- [ ] Code starts from google research's maskgit and lucidrain's maskgit-muse repo
