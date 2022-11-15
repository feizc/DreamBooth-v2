import argparse 
import os 
from pathlib import Path

import torch 
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image 


from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel



def parse_args():
    parser = argparse.ArgumentParser(description="training script for dreambooth") 
    
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    args = parser.parse_args()
    return args 


class DreamBoothDataset(Dataset): 
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size 
        self.center_crop = center_crop 
        self.tokenizer = tokenizer 
        self.instance_data_root = Path(instance_data_root) 

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images 







def main():
    args = parse_args()



if __name__ == '__main__': 
    main()