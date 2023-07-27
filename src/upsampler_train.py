import flax
import jax
import jax.numpy as jnp
import wandb
from models import vqgan_tokenizer, maskgit_transformer
from configs import maskgit_class_cond_config
from utils import restore_from_path
import jax.numpy as jnp
import math

from utils import cosine_schedule

from models.generator import ImageNet_class_conditional_generator


from libml.losses import  sigmoid_cross_entropy_with_logits

from dataloaders.upsampler_text import get_dataset


from torch.utils.data import DataLoader

noise_schedules = {
    "cosine": cosine_schedule
}

config = dict(
    NUM_EPOCHS = 100,
    SAVE_INTERVAL = 1,
    BATCH_SIZE = 8,
    NOISE_SCHEDULE = "cosine",
)

noise_schedule = noise_schedules[config["NOISE_SCHEDULE"]]

# Initialize WandB
wandb.init(project="maskgit-training", config=config)

model = ImageNet_class_conditional_generator(config)

transformer_model = model.transformer_model
tokenizer_model = model.tokenizer_model
# Create the dataset and dataloader
dataset = get_dataset()
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# Create optimizer and scheduler
optimizer = flax.optim.Adam(learning_rate=0.001).create(transformer_model)
# scheduler = ...  


# Training Loop
for epoch in range(config.NUM_EPOCHS):
    step = 0
    for batch in dataloader:
        should_log = True
        step += 1
        
        # Extract 512x512 image and upscaled 256x256 image from the batch
        # Upscaled image will be used as label
        images, labels = batch['image'], batch['upscaled_image']

        # Assert that they are images
        assert jnp.issubdtype(images.dtype, jnp.floating), 'Images are not in the expected format'
        assert jnp.issubdtype(labels.dtype, jnp.floating), 'Labels are not in the expected format'

        # Encode the images to get image tokens
        with jax.lax.stop_gradient():
            _, images, _ = tokenizer_model.encode(images)
            _, labels, _ = tokenizer_model.encode(labels)
        
        # Define loss function based on the provided inspiration
        result_images = None

        def loss_fn(params, model, images, labels):
            # 1. Tokenizing Images
            if jnp.issubdtype(images.dtype, jnp.floating):
                with jax.lax.stop_gradient():
                    _, ids, _ = tokenizer_model.encode(images)  # Adjust this line accordingly based on your tokenizer's method signature.
            else:
                ids = images
            
            # We won't handle conditioning images for simplicity, but you can follow the same logic as in PyTorch code.

            # 3. Masking for Training
            batch, seq_len = ids.shape
            rand_time = jnp.random.uniform((batch,))
            rand_mask_probs = noise_schedule(rand_time)  # Ensure you have an equivalent function in Jax.
            num_token_masked = (seq_len * rand_mask_probs).round().clip(min = 1)
            
            # Jax doesn't have argsort for random values like PyTorch does.
            # So, we might need a custom function or a workaround to generate the mask.
            # Here's a simple workaround:
            ranks = jax.ops.segment_sum(jnp.ones_like(ids), ids, num_segments=seq_len)
            mask = ranks < jax.lax.broadcast_in_dim(num_token_masked, (batch, seq_len), (0,))
            
            ignore_index = -1
            masked_labels = jnp.where(mask, ids, ignore_index)
            
            logits = model.generate_logits(params, ids, return_logits=True, labels=masked_labels, return_images=True)
            return sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

        # Forward and Backward pass
        loss, grads = jax.value_and_grad(loss_fn)(optimizer.target, transformer_model, images, labels)
        optimizer = optimizer.apply_gradient(grads)
        # optimizer = scheduler(optimizer, epoch)  # Update LR if scheduler is used

        # Logging with WandB
        wandb.log({"loss": loss})

        # Optionally save intermediate results
        if result_images is not None:
            images = model.p_generate_samples()
            wandb.log({"Generated Images": [wandb.Image(img) for img in result_images]})


# Save final model checkpoint
wandb.save("./path_to_save_checkpoint")
