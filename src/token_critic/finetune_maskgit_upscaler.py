# This script will: 
# 1. Create the VQ-GAN model and load the checkpoint
# 2. Create the MaskGITTransofmer model and load the checkpoint
# 3. Create the dataset and dataloader
# 4. Create the optimizer and scheduler
# 5. Train the model on 256 x 512 paired images
#     - save samples every loop with wandb.Image

import jax.numpy as jnp
from flax import linen as nn

# Define utility function to check existence, similar to PyTorch's `exists` function
def exists(val):
    return val is not None

# Define rearrange utility function
def rearrange(x):
    return jnp.squeeze(x, axis=-1)

class SelfCritic(nn.Module):
    net: nn.Module
    dim: int
    
    def setup(self):
        self.to_pred = nn.Dense(features=1)
    
    def forward_with_cond_scale(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_cond_scale(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)

    def forward_with_neg_prompt(self, x, *args, **kwargs):
        _, embeds = self.net.forward_with_neg_prompt(x, *args, return_embed=True, **kwargs)
        return self.to_pred(embeds)

    def __call__(self, x, *args, labels=None, **kwargs):
        _, embeds = self.net(x, *args, return_embed=True, **kwargs)
        logits = self.to_pred(embeds)
        
        if not exists(labels):
            return logits
        
        logits = rearrange(logits)
        return jnp.mean(nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
