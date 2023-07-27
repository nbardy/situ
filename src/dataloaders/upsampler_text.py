import numpy as np
from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import torch

from typing import Dict
# from open_clip_jax import tokenize

import numpy as np
from PIL import Image

from transformers import AutoTokenizer, FlaxCLIPModel

clip_model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


# 1. Custom PyTorch Dataset


class AllCropsStreamingSuperResolutionDataset(Dataset):
    def __init__(self, width, height, crops, subject_crop, downsize_method, flatten_crops, upscale_factor, with_text=False):
        self.dataset = load_dataset(
            "laion/laion-high-resolution", streaming=True)
        self.width = width
        self.height = height
        self.crops = crops
        self.subject_crop = subject_crop
        self.downsize_method = downsize_method
        self.flatten_crops = flatten_crops
        self.upscale_factor = upscale_factor
        self.with_text = with_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        entry = self.dataset[idx]
        image_path = entry['URL']
        image = Image.open(image_path).convert('RGB')

        if self.subject_crop:
            # TODO: Implement subject-based cropping logic (e.g., using a detector)
            pass

        crops = self._generate_center_biased_crops(
            image, self.width, self.height, self.crops)

        text_embedding = None
        # Tokenizing the text
        if self.with_text is True:
            text = entry['TEXT']
            # text_embedding = tokenize([text])._numpy()
            # image = np.expand_dims(jax_preprocess(Image.open("CLIP.png")), 0)
            # image_embed = image_fn(jax_params, image)
            inputs = clip_tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
            text_embedding = clip_model.get_text_features(**inputs)



        # Use transform function to generate the different image versions
        image_input_small, image_input_full, image_input_rescaled_full = self._image_processing(
            crops[0])

        if self.flatten_crops:
            crops = [F.to_tensor(crop) for crop in crops]

        return {
            'text': text,
            'text_embedding': text_embedding,
            'image_input_small': image_input_small,
            'image_input_full': image_input_full,
            'image_input_rescaled_full': image_input_rescaled_full,
            'crops': crops
        }

    def _generate_center_biased_crops(self, image, width, height, num_crops):
        im_w, im_h = image.size

        # Center of the image
        center_x, center_y = im_w // 2, im_h // 2

        # Generate crops biased towards the center
        crops = []
        for _ in range(num_crops):
            # Random offset biased towards the center
            off_x = int(np.random.normal(center_x, im_w * 0.1))
            off_y = int(np.random.normal(center_y, im_h * 0.1))

            # Making sure it's within bounds
            off_x = np.clip(off_x, 0, im_w - width)
            off_y = np.clip(off_y, 0, im_h - height)

            crops.append(image.crop(
                (off_x, off_y, off_x + width, off_y + height)))

        return crops


    def _image_processing(self, image):
        # Placeholder - you need to specify the transformations
        image_input_small = F.to_tensor(F.resize(image, (256, 256)))
        image_input_full = F.to_tensor(image)
        image_input_rescaled_full = F.to_tensor(F.resize(image, (self.width, self.height)))
        return image_input_small, image_input_full, image_input_rescaled_full

def collate_fn(batch):
    # Flattening crops
    flat_crops = [item['crops']
                    for sublist in batch for item in sublist['crops']]

    # Duplicating other fields
    fields = ['text', 'text_embedding', 'image_input_small',
                'image_input_full', 'image_input_rescaled_full']
    collated_batch = {field: [item[field] for item in batch for _ in range(
        len(batch[0]['crops']))] for field in fields}

    collated_batch['crops'] = torch.stack(flat_crops)
    return collated_batch


def get_dataset():
    all_crops_dataset = AllCropsStreamingSuperResolutionDataset(
        flatten_crops=True, width=512, height=512, upscale_factor=4)
    flat = DataLoader(
        all_crops_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    return flat
