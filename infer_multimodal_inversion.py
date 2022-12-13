import os
import torch
import argparse 
from torch import autocast

import PIL
from PIL import Image
import random 
from tqdm import tqdm 

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    
    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
    
    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds



@torch.no_grad()
def stablediffusion(vae, unet, clip_tokenizer, clip, device, prompt="", guidance_scale=7.5, steps=50, seed=None, width=512, height=512, ):
    
    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: 
        seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)
    
    #Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps)
    
    init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
    t_start = 0
    
    #Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    #latent = noise * scheduler.init_noise_sigma
    latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=device)).to(device)
    
    #Process clip
    with autocast(device):
        tokens_unconditional = clip_tokenizer("", padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

            
        timesteps = scheduler.timesteps[t_start:]
        
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + i

            #sigma = scheduler.sigmas[t_index]
            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            #Predict the unconditional noise residual
            noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
                
            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
                
            #Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latent = scheduler.step(noise_pred, t_index, latent).prev_sample

        #scale and decode the image latents with vae
        latent = latent / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)





def main():
    model_path = './ckpt' 
    learned_embeds_path = './save/learned_embeds.bin'  
    learned_image_path = './save/image_embeds.bin'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(model_path, 'tokenizer')
    )
    text_encoder = CLIPTextModel.from_pretrained(
        os.path.join(model_path, 'text_encoder')
    )
    load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer)

    learned_image_embeds = torch.load(learned_image_path, map_location="cpu")
    learned_image_embeds = learned_image_embeds['image_inversion']

    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
    ).to(device) 

    prompt = 'city under the sun, painting, in a style of <sks>' 

    num_samples = 2 #@param {type:"number"}
    num_rows = 2 #@param {type:"number"}

    all_images = [] 
    for _ in range(num_rows):
        with autocast("cuda"):
            images = pipe.combine_forward(image_latents=learned_image_embeds, combine_strength=0.5, prompt=prompt, num_images_per_prompt=num_samples, num_inference_steps=50, guidance_scale=7.5).images
        all_images.extend(images)

    grid = image_grid(all_images, num_samples, num_rows) 
    grid.save('./1.png')




if __name__ == '__main__': 
    main()
