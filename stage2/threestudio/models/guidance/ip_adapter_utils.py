from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    
)
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union
from ip_adapter import IPAdapter


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

import inspect

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
  
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="2.1",
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        
        
        # noise_scheduler = DDIMScheduler(
        #     num_train_timesteps=1000,
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="scaled_linear",
        #     clip_sample=False,
        #     set_alpha_to_one=False,
        #     steps_offset=1,
        # )
        self.dtype = torch.float16 if fp16 else torch.float32
        noise_scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler", torch_dtype=self.dtype
        )        
        self.scheduler=noise_scheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            # revision="fp16",
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(dtype=torch.float16).to('cuda'),
            local_files_only=True,
            feature_extractor=None,
            safety_checker=None,            
        ).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        # pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
        self.ip_model = IPAdapter(pipe, '/root/autodl-tmp/IP-Adapter/IPAdapter/models/image_encoder', '/root/autodl-tmp/IP-Adapter/IPAdapter/models/ip-adapter_sd15.bin', device)
        pipe=self.ip_model.pipe
        

        

        # Create model
        # pipe = StableDiffusionPipeline.from_pretrained(
        #     model_key, torch_dtype=self.dtype
        # )
        # pipe = StableDiffusionPipeline.from_pretrained('/root/autodl-tmp/GaussianEditor/TextualInversion/dreambooth-concept'
        #     ,local_files_only=True,torch_dtype=self.dtype,
        # )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        # self.scheduler = DDIMScheduler.from_pretrained(
        #     model_key, subfolder="scheduler", torch_dtype=self.dtype
        # )

        # del pipe
        self.pipe=pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = {}
        self.clip_image_processor = CLIPImageProcessor()

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings['pos'] = pos_embeds
        self.embeddings['neg'] = neg_embeds

        # # directional embeddings
        # for d in ['front', 'side', 'back']:
        #     embeds = self.encode_text([f'{p}, {d} view' for p in prompts])
        #     self.embeddings[d] = embeds
    
    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    

    
    def train_step(
        self,
        pred_rgb,
        pil_image, # prompt image
        step_ratio=None,
        guidance_scale=7.5,
        as_latent=False,
        vers=None, hors=None,
    ):
        num_samples=pred_rgb.shape[0]
        pred_rgb = pred_rgb.permute(0, 3, 1, 2)
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        
        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)        
        
        
        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_model.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )        
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)        

        # self.ip_model
        # with torch.inference_mode():
        prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
            ["best quality, high quality"],
            device=self.device,
            num_images_per_prompt=num_samples,
            do_classifier_free_guidance=True,
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"],
        )
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        t = torch.randint(self.min_step,self.max_step + 1,[batch_size],dtype=torch.long,device=self.device)            

        noise = torch.randn_like(latents)  # TODO: use torch generator
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        
        noise_pred = self.unet(
            latent_model_input,
            torch.cat([t] * 2),
            encoder_hidden_states=prompt_embeds,
        ).sample
        
        # perform guidance (high scale from paper!)
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # seems important to avoid NaN...
        # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss
        




    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents



    def train_step_old(
        self,
        pred_rgb,
        pil_image,
        step_ratio=None,
        guidance_scale=7.5,
        as_latent=False,
        vers=None, hors=None,
    ):
        pred_rgb = pred_rgb.permute(0, 3, 1, 2)
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
        
        # num_samples=1
        num_samples=pred_rgb.shape[0]
        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_model.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )        
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)        

        
        
        # self.embeddings['pos'] = torch.cat([self.embeddings['pos'], image_prompt_embeds], dim=1)
        # self.embeddings['neg'] = torch.cat([self.embeddings['neg'], uncond_image_prompt_embeds], dim=1)

        pos_embed = torch.cat([self.embeddings['pos'], image_prompt_embeds], dim=1)
        neg_embed = torch.cat([self.embeddings['neg'], uncond_image_prompt_embeds], dim=1)
                

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        with torch.no_grad():
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            else:
                t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

            # predict the noise residual with unet, NO grad!
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            if hors is None:
                # embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
                embeddings = torch.cat([pos_embed.expand(batch_size, -1, -1), neg_embed.expand(batch_size, -1, -1)])
            else:
                def _get_dir_ind(h):
                    if abs(h) < 60: return 'front'
                    elif abs(h) < 120: return 'side'
                    else: return 'back'

                embeddings = torch.cat([self.embeddings[_get_dir_ind(h)] for h in hors] + [self.embeddings['neg'].expand(batch_size, -1, -1)])

            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=embeddings
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            # seems important to avoid NaN...
            # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss
    
    @torch.no_grad()
    def refine(self, pil_image,pred_rgb,
               guidance_scale=7.5, steps=100, strength=0.05,
        ):
        pred_rgb = pred_rgb.permute(0, 3, 1, 2)
        
        # torch.save(pred_rgb,'./pred_rgb.png')
        batch_size = pred_rgb.shape[0]
        num_samples = pred_rgb.shape[0]
        
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # pred_rgb_512=pred_rgb
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)
        
        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_model.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )        
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)        
        
        pos_embed = torch.cat([self.embeddings['pos'], image_prompt_embeds], dim=1)
        neg_embed = torch.cat([self.embeddings['neg'], uncond_image_prompt_embeds], dim=1)
        

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        # embeddings = torch.cat([self.embeddings['pos'].expand(batch_size, -1, -1), self.embeddings['neg'].expand(batch_size, -1, -1)])
        embeddings = torch.cat([pos_embed.expand(batch_size, -1, -1), neg_embed.expand(batch_size, -1, -1)])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings,
            ).sample

            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        imgs=F.interpolate(imgs, (pred_rgb.shape[2], pred_rgb.shape[3]), mode='bilinear', align_corners=False)
        return imgs    
    
from diffusers.utils import load_image
from torchvision.transforms.functional import pil_to_tensor,to_pil_image
from torchvision.utils import save_image
if __name__ == "__main__":
    sd=StableDiffusion(torch.device('cuda'))
    sd.get_text_embeds(['best quality, high quality'], ["deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality"])
    img1=load_image('/root/autodl-tmp/GaussianEditor/edit_cache_Copy/-root-autodl-tmp-gaussian-splatting-output-d8afce2d-6-point_cloud-iteration_30000-point_cloud.ply/IDM_Render/0000.jpg')
    img2=load_image('/root/autodl-tmp/GaussianEditor/edit_cache_Copy/-root-autodl-tmp-gaussian-splatting-output-d8afce2d-6-point_cloud-iteration_30000-point_cloud.ply/IDM_Render/0007.jpg')
    img1=pil_to_tensor(img1).unsqueeze(0).cuda().float()
    img2=pil_to_tensor(img2).unsqueeze(0).cuda().float()
    
    save_image(sd.refine(pil_image=img1,pred_rgb=img2),'./refined_new.png')