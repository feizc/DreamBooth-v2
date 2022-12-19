# Multimodal Inversion


## What does this repo do?
We aim to personalize text-to-image generation with multimodal inversion
As original text inversion struggles with learning precise shapes for objective, here we introduce to use additional learnable image latents to help make a better detail controling. 

We make the following points:
1. Randomness: learn the mean and variance of image latents, and sample an image latents each time; 
2. Two stage: first train the textual inversion well, then optimize the image latents;  

Our implementation is totally compatible with [diffusers](https://github.com/huggingface/diffusers) and stable diffusion model.

## Training 

To training the multimodal inversion on the corresponding image dataset, run:
```
python train_multimodal_inversion_two_stage.py
```

The trained text and image embedding will be saved in the corresponding file path. 

## Infer

To evaluate the performance of multimodal inversion systerm, run:
```
python infer_multimodal_inversion.py
```

Moreover, we use a hyper-parameter combine_strength to control the incorporation of image latents to noise initialization referring to [img2img](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py). 

