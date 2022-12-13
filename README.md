# Personalizing Text-to-Image Generation with Multimodal Inversion

As we find that original text inversion struggles with learning precise shapes for objective, we introduce to use learnable image latents to help make a better detail controling. 

We make the following points:
1. Randomness: learn the mean and variance of image latents, and sample a image latents each time; 
2. Two stage: first train the textual inversion well, then optimize the image latents;  


