import torch
from optimum.onnxruntime import ORTStableDiffusionPipeline
import time
from diffusers import DiffusionPipeline, AutoencoderTiny

seed = 42
st = time.time()
# Load model.
pipe = ORTStableDiffusionPipeline.from_pretrained('../models/sdxs-512-0.9-onnx')
vae = AutoencoderTiny.from_pretrained("../models/sdxs-512-0.9/vae")
print(f'load {time.time() - st}')

# pipe.vae = AutoencoderKL.from_pretrained("IDKiro/sdxs-512-0.9/vae_large")     # use original VAE
st = time.time()
prompt =  "dune, planet, novel graphic"
# Ensure using the same inference steps as the loaded model and CFG set to 0.
latents = pipe(prompt,
            height=128*3,
            width=128*2,
            num_inference_steps=1,
            guidance_scale=1.0,
            output_type="latent").images #.images[0]
with torch.no_grad():
    latents = vae.decode(torch.from_numpy(latents) / vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * latents.shape[0]
    image = pipe.image_processor.postprocess(latents.numpy(), output_type='pil', do_denormalize=do_denormalize)[0]
image.save("./output.png")
print(f'done {time.time() - st}')