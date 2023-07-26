import flax
import jax
import jax.numpy as jnp
import wandb
from maskgit.nets import vqgan_tokenizer, maskgit_transformer
from maskgit.configs import maskgit_class_cond_config
from maskgit.utils import restore_from_path

from libml.losses import  sigmoid_cross_entropy_with_logits

from dataloaders.upsampler_text import get_dataset


from torch.utils.data import DataLoader

NUM_EPOCHS = 100    
SAVE_INTERVAL = 1

# Initialize WandB
wandb.init(project="maskgit-training")

# Load Model Configurations
maskgit_cf = maskgit_class_cond_config.get_config()

# Create VQ-GAN model and load the checkpoint
tokenizer_model = vqgan_tokenizer.VQVAE(config=maskgit_cf, dtype=jnp.float32, train=False)
tokenizer_variables = restore_from_path("./path_to_tokenizer_checkpoint")

# Create MaskGITTransofmer model and load the checkpoint
transformer_model = maskgit_transformer.Transformer(...)
transformer_variables = restore_from_path("./path_to_maskgit_checkpoint")

# Create the dataset and dataloader
dataset = get_dataset()
dataloader = DataLoader(dataset, batch_size=maskgit_cf.eval_batch_size, shuffle=True)

# Create optimizer and scheduler
optimizer = flax.optim.Adam(learning_rate=0.001).create(transformer_model)
scheduler = ...  # Add your scheduler if needed

# Define loss function based on the provided inspiration
def loss_fn(params, model, images, labels):
    logits, _ = model.apply(params, images, return_logits=True, labels=labels)
    return sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

# Training Loop
for epoch in range(NUM_EPOCHS):
    step = 0
    for batch in dataloader:
        step += 1
        # Extract 512x512 image and upscaled 256x256 image from the batch
        # Upscaled image will be used as label
        images, labels = batch['image'], batch['upscaled_image']

        # Forward and Backward pass
        loss, grads = jax.value_and_grad(loss_fn)(optimizer.target, transformer_model, images, labels)
        optimizer = optimizer.apply_gradient(grads)
        optimizer = scheduler(optimizer, epoch)  # Update LR if scheduler is used

        # Logging with WandB
        wandb.log({"Loss": loss})

        # Optionally save intermediate results
        if step % SAVE_INTERVAL == 0:
            gen_images = transformer_model.apply(transformer_variables, images, method=transformer_model.generate_samples)
            wandb.log({"Generated Images": [wandb.Image(img) for img in gen_images]})

# Save final model checkpoint
wandb.save("./path_to_save_checkpoint")
