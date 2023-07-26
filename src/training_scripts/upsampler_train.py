import flax
import jax
import jax.numpy as jnp
import wandb
from nets import vqgan_tokenizer, maskgit_transformer
from configs import maskgit_class_cond_config
from utils import restore_from_path

from models.trained import ImageNet_class_conditional_generator


from libml.losses import  sigmoid_cross_entropy_with_logits

from dataloaders.upsampler_text import get_dataset


from torch.utils.data import DataLoader

config = dict(
    NUM_EPOCHS = 100,
    SAVE_INTERVAL = 1,
    BATCH_SIZE = 8,
)

# Initialize WandB
wandb.init(project="maskgit-training")

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

        # Define loss function based on the provided inspiration
        result_images = None

        # TODO: Add token critic
        def loss_fn(params, model, images, labels):
            logits = model.generate_logits(params, images, return_logits=True, labels=labels, return_images=True)
            return sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

        # Forward and Backward pass
        loss, grads = jax.value_and_grad(loss_fn)(optimizer.target, transformer_model, images, labels)
        optimizer = optimizer.apply_gradient(grads)
        # optimizer = scheduler(optimizer, epoch)  # Update LR if scheduler is used

        # Logging with WandB
        wandb.log({"loss": loss})

        # Optionally save intermediate results
        if result_images is not None:
            wandb.log({"Generated Images": [wandb.Image(img) for img in result_images]})


# Save final model checkpoint
wandb.save("./path_to_save_checkpoint")
