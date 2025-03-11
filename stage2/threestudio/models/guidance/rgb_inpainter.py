import time
from dataclasses import dataclass

import mediapy
import torch
import wandb
from diffusers import (
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
    AutoPipelineForInpainting,
)
from diffusers.utils.torch_utils import randn_tensor
from nerfiller.guidance.feature_extractor import FeatureExtractor
from nerfiller.guidance.multiview_metric import MultiviewMetric
from nerfiller.utils.camera_utils import rescale_intrinsics
from nerfiller.utils.diff_utils import (
    encode_prompt,
    get_decoder_approximation,
    tokenize_prompt,
    get_epsilon_from_v_prediction,
    get_v_prediction_from_epsilon,
)
from nerfiller.utils.grid_utils import make_grid, undo_grid
from nerfiller.utils.misc import cleanup
from nerfiller.utils.typing import *
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from nerfiller.utils.image_utils import save_video_from_path

from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDPMScheduler
)
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)

from transformers import CLIPTextModel

CONSOLE = Console()


@dataclass
class ModelInput:
    """Input for Inpainting Model."""

    latents: Float[Tensor, "B 4 H W"]
    latents_mask: Float[Tensor, "B 1 H W"]
    masked_image_latents: Float[Tensor, "B 4 H W"]
    latents_mask_uncond: Float[Tensor, "B 1 H W"]
    """This is an image of all 1s."""
    masked_image_latents_uncond: Float[Tensor, "B 4 H W"]
    """This is an image of all 0s."""
    noise: Float[Tensor, "B 4 H W"]


class RGBInpainter:
    """
    Module for inpainting with the stable diffusion inpainting pipeline.
    """

    def __init__(
        self,
        half_precision_weights: bool = True,
        lora_model_path: Optional[str] = None,
        device: str = "cuda:0",
        vae_device: str = "cuda:0",
        pipeline_name: str = "stabilityai/stable-diffusion-2-inpainting",
    ):
        print(f"Loading RGB Inpainter ...")

        self.half_precision_weights = half_precision_weights
        self.lora_model_path = lora_model_path
        self.device = device
        self.vae_device = vae_device
        self.dtype = torch.float16 if self.half_precision_weights else torch.float32
        self.pipeline_name = pipeline_name
        self.set_pipe()
        self.setup()

    def set_pipe(self):
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.dtype,
        }
        # self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     self.pipeline_name,
        #     **pipe_kwargs,
        # )
        
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
            revision=None,
            vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse'),
        )

        pipe.unet = UNet2DConditionModel.from_pretrained(
            # '/root/autodl-tmp/Lora/realfill/IDM-model', subfolder="unet", revision=None,
            # '/root/autodl-tmp/Lora/realfill/cloth-model', subfolder="unet", revision=None,
            '/root/autodl-tmp/Lora/realfill/pipeline-model', subfolder="unet", revision=None,
            
        )
        pipe.text_encoder = CLIPTextModel.from_pretrained(
            # '/root/autodl-tmp/Lora/realfill/IDM-model', subfolder="text_encoder", revision=None,
            # '/root/autodl-tmp/Lora/realfill/cloth-model', subfolder="text_encoder", revision=None,
            '/root/autodl-tmp/Lora/realfill/pipeline-model', subfolder="text_encoder", revision=None,
            
        )
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")        
        
        self.pipe=pipe

    def setup(self):
        # Load LoRA
        if self.lora_model_path:
            self.pipe.load_lora_weights(self.lora_model_path)
            print(f"Loaded LoRA model from {self.lora_model_path}")

        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device).eval()

        self.unet = self.pipe.unet.to(self.device).eval()
        self.vae = self.pipe.vae.to(self.vae_device).eval()

        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        self.vae_latent_channels = self.pipe.vae.config.latent_channels

        # self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        del self.pipe
        cleanup()

        print(f"Loaded RGB inpainter!")

    def compute_text_embeddings(self, prompt: str, negative_prompt: str):
        """Get the text embeddings for a string."""
        assert self.tokenizer is not None
        assert self.text_encoder is not None
        with torch.no_grad():
            text_inputs = tokenize_prompt(self.tokenizer, prompt, tokenizer_max_length=None)
            prompt_embeds = encode_prompt(
                self.text_encoder,
                text_inputs.input_ids,
                text_inputs.attention_mask,
                text_encoder_use_attention_mask=False,
            )
            negative_text_inputs = tokenize_prompt(self.tokenizer, negative_prompt, tokenizer_max_length=None)
            negative_prompt_embeds = encode_prompt(
                self.text_encoder,
                negative_text_inputs.input_ids,
                negative_text_inputs.attention_mask,
                text_encoder_use_attention_mask=False,
            )

        return [prompt_embeds, negative_prompt_embeds]

    def destroy_text_encoder(self) -> None:
        """Delete the text modules to save on memory."""
        del self.tokenizer
        del self.text_encoder
        cleanup()

    def forward_unet(
        self,
        sample,
        t,
        text_embeddings,
        denoise_in_grid: bool = False,
    ):
        # process embeddings
        prompt_embeds, negative_prompt_embeds = text_embeddings

        batch_size = sample.shape[0] // 3

        prompt_embeds = torch.cat(
            [
                prompt_embeds.repeat(batch_size, 1, 1),
                negative_prompt_embeds.repeat(batch_size, 1, 1),
                negative_prompt_embeds.repeat(batch_size, 1, 1),
            ]
        )

        if denoise_in_grid:
            grid_sample = make_grid(sample)
            grid_prompt_embeds = prompt_embeds[:3].repeat(grid_sample.shape[0] // 3, 1, 1)
            noise_pred = self.unet(
                sample=grid_sample,
                timestep=t,
                encoder_hidden_states=grid_prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = undo_grid(noise_pred)
        else:
            noise_pred = self.unet(
                sample=sample,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        return noise_pred

    def get_noise_pred(
        self,
        t,
        model_input: ModelInput,
        text_embeddings,
        text_guidance_scale: float = 0.0,
        image_guidance_scale: float = 0.0,
        denoise_in_grid: bool = False,
        multidiffusion_steps: int = 1,
        multidiffusion_type: str = "epsilon",
        randomize_latents: bool = False,
        randomize_within_grid: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        only_noise_pred: bool = False,
    ):
        assert self.scheduler.config.prediction_type == "epsilon", "We assume the model predicts epsilon."

        batch_size = model_input.latents.shape[0]
        value = torch.zeros_like(model_input.latents)
        count = torch.zeros_like(model_input.latents)

        for i in range(multidiffusion_steps):
            if randomize_latents:
                indices = torch.randperm(batch_size)
            else:
                indices = torch.arange(batch_size)

            if denoise_in_grid and randomize_within_grid:
                for j in range(0, len(indices), 4):
                    indices[j : j + 4] = indices[j : j + 4][torch.randperm(4)]

            latents = model_input.latents[indices]
            latents_mask = model_input.latents_mask[indices]
            latents_mask_uncond = model_input.latents_mask_uncond[indices]
            masked_image_latents = model_input.masked_image_latents[indices]
            masked_image_latents_uncond = model_input.masked_image_latents_uncond[indices]

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents, latents, latents])
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            latents_mask_input = torch.cat([latents_mask, latents_mask, latents_mask_uncond])
            masked_image_latents_input = torch.cat(
                [
                    masked_image_latents,
                    masked_image_latents,
                    masked_image_latents_uncond,
                ]
            )

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input_cat = torch.cat(
                [latent_model_input, latents_mask_input, masked_image_latents_input],
                dim=1,
            )

            # TODO: save compute by skipping some text encodings if not using them in CFG

            noise_pred_all = self.forward_unet(
                sample=latent_model_input_cat,
                t=t,
                text_embeddings=text_embeddings,
                denoise_in_grid=denoise_in_grid,
            )

            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred_all.chunk(3)

            noise_pred = (
                noise_pred_image
                + text_guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            if multidiffusion_type == "v_prediction":
                v_prediction = get_v_prediction_from_epsilon(noise_pred, t, latents, self.scheduler.alphas_cumprod)
                value[indices] += v_prediction
                count[indices] += 1
            elif multidiffusion_type == "epsilon":
                value[indices] += noise_pred
                count[indices] += 1
            else:
                raise ValueError("Not implemented.")

        # take the MultiDiffusion step
        final_noise_pred = torch.where(count > 0, value / count, value)

        if multidiffusion_type == "v_prediction":
            final_noise_pred = get_epsilon_from_v_prediction(
                final_noise_pred,
                t.item(),
                model_input.latents,
                self.scheduler.alphas_cumprod,
            )
        elif multidiffusion_type == "epsilon":
            pass
        else:
            raise ValueError("Not implemented.")

        if only_noise_pred:
            return None, None, final_noise_pred

        scheduler_output = self.scheduler.step(final_noise_pred, t, model_input.latents, generator=generator)
        pred_prev_sample = scheduler_output.prev_sample
        pred_original_sample = scheduler_output.pred_original_sample

        assert not pred_prev_sample.isnan().any()
        assert not pred_original_sample.isnan().any()
        return pred_prev_sample, pred_original_sample, final_noise_pred

    def get_model_input(
        self,
        image: Float[Tensor, "B 3 H W"],
        mask: Float[Tensor, "B 1 H W"],
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        starting_image: Optional[Float[Tensor, "B 3 H W"]] = None,
        starting_timestep: Optional[int] = None,
        keep_grad: bool = False,
    ) -> ModelInput:
        """Returns the inputs for the unet."""

        # TODO: incorporate seeds

        batch_size, _, height, width = image.shape

        noise = randn_tensor(
            shape=(
                batch_size,
                self.vae_latent_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            ),
            generator=generator,
            device=torch.device(self.device),
            dtype=self.dtype,
        )
        if starting_image is not None:
            assert starting_timestep is not None
            if keep_grad:
                latents = self.encode_images(starting_image)
            else:
                with torch.no_grad():
                    latents = self.encode_images(starting_image)
            latents = self.scheduler.add_noise(latents, noise, starting_timestep)
        else:
            latents = noise

        latents_mask = torch.nn.functional.interpolate(
            mask,
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="nearest",
        )
        print('len(torch.unique(latents_mask))',len(torch.unique(latents_mask)))
        assert len(torch.unique(latents_mask)) <= 2
        latents_mask = latents_mask.to(device=self.device, dtype=self.dtype)
        assert len(torch.unique(mask)) <= 2
        masked_image = torch.where(mask == 0, image, 0.5)
        with torch.no_grad():
            masked_image_latents = self.encode_images(masked_image)

        latents_mask_uncond = torch.ones_like(latents_mask)
        masked_image_uncond = torch.ones_like(masked_image) * 0.5
        with torch.no_grad():
            masked_image_latents_uncond = self.encode_images(masked_image_uncond)

        model_input = ModelInput(
            latents.to(device=self.device, dtype=self.dtype),
            latents_mask.to(device=self.device, dtype=self.dtype),
            masked_image_latents.to(device=self.device, dtype=self.dtype),
            latents_mask_uncond.to(device=self.device, dtype=self.dtype),
            masked_image_latents_uncond.to(device=self.device, dtype=self.dtype),
            noise.to(device=self.device, dtype=self.dtype),
        )

        return model_input

    def get_loss(
        self,
        x0: Float[Tensor, "B 3 H W"],
        mask: Float[Tensor, "B 1 H W"],
        depth: Optional[Float[Tensor, "B 1 H W"]] = None,
        multiview_guidance_scale: float = 0.0,
        reconstruction_guidance_scale: float = 0.0,
        feature_extractor: Optional[FeatureExtractor] = None,
        multiview_metric: Optional[MultiviewMetric] = None,
        K: Optional[Float[Tensor, "B 3 3"]] = None,
        c2w: Optional[Float[Tensor, "B 3 4"]] = None,
        output_folder: Optional[Path] = None,
        step: int = 0,
        guidance_step: int = 0,
        starting_image: Optional[Float[Tensor, "B 3 H W"]] = None,
    ):
        """Losses on the VAE decoded images x0.
        The multi-view loss is applied where mask == 0.0 (regions that have known depth).
        """

        loss = 0.0

        if multiview_guidance_scale != 0.0:
            features = feature_extractor(x0.to(feature_extractor.device)).to(self.device)

            # multiview guidance
            scale_factor = features.shape[-1] / x0.shape[-1]
            K_scaled = rescale_intrinsics(K, scale_factor, scale_factor)
            mask_scaled = 1.0 - torch.nn.functional.interpolate(mask, scale_factor=scale_factor, mode="nearest")
            depth_scaled = torch.nn.functional.interpolate(depth, scale_factor=scale_factor, mode="bilinear")
            for cam1 in range(len(c2w)):
                for cam2 in range(cam1 + 1, len(c2w)):
                    loss_mv, loss_dict = multiview_metric(
                        features1=features[cam1 : cam1 + 1],
                        features2=features[cam2 : cam2 + 1],
                        K1=K_scaled[cam1 : cam1 + 1],
                        K2=K_scaled[cam2 : cam2 + 1],
                        c2w1=c2w[cam1 : cam1 + 1],
                        c2w2=c2w[cam2 : cam2 + 1],
                        image1=x0[cam1 : cam1 + 1],
                        image2=x0[cam2 : cam2 + 1],
                        mask1=mask_scaled[cam1 : cam1 + 1],
                        mask2=mask_scaled[cam2 : cam2 + 1],
                        depth1=depth_scaled[cam1 : cam1 + 1],
                        depth2=depth_scaled[cam2 : cam2 + 1],
                        output_folder=output_folder if (cam1 == 0 and guidance_step == 0) else None,
                        suffix=f"-{step:06d}-{cam1:06d}-{cam2:06d}-{guidance_step:06d}",
                    )
                    loss += multiview_guidance_scale * loss_mv.sum()

        if reconstruction_guidance_scale != 0.0:
            loss += (
                reconstruction_guidance_scale * (((starting_image.to(x0.device) - x0) * mask.to(x0.device)) ** 2).mean()
            )

        return loss

    @torch.cuda.amp.autocast(enabled=True)
    def get_image(
        self,
        text_embeddings,
        image: Float[Tensor, "B 3 H W"],
        mask: Float[Tensor, "B 1 H W"],
        num_inference_steps: int = 20,
        denoise_in_grid: bool = False,
        depth: Optional[Float[Tensor, "B 1 H W"]] = None,
        text_guidance_scale: Optional[float] = None,
        image_guidance_scale: Optional[float] = None,
        multidiffusion_steps: int = 1,
        multidiffusion_type: str = "epsilon",
        randomize_latents: bool = False,
        randomize_within_grid: bool = False,
        use_decoder_approximation: bool = False,
        multiview_guidance_scale: float = 0.0,
        reconstruction_guidance_scale: float = 0.0,
        feature_extractor: Optional[FeatureExtractor] = None,
        multiview_metric: Optional[MultiviewMetric] = None,
        K: Optional[Float[Tensor, "B 3 3"]] = None,
        c2w: Optional[Float[Tensor, "B 3 4"]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        show_multiview: bool = False,
        guidance_steps: List[int] = [5],
        num_guidance_steps: int = 10,
        classifier_guidance_scale: float = 0.0,
        output_folder: Optional[Path] = None,
        starting_image: Optional[Float[Tensor, "B 3 H W"]] = None,
        starting_lower_bound: Optional[float] = None,
        starting_upper_bound: Optional[float] = None,
        classifier_guidance_loss_rescale=1000.0,
        classifier_guidance_start_step: int = 0,
        replace_original_pixels: bool = False,
    ) -> Float[Tensor, "B 3 H W"]:
        """Run the denoising sampling process, also known as the reverse process.
        Inpaint where mask == 1.
        If output folder is not None, then save images to this folder.

        Args:
            text_embeddings: Either 2 per image (BB) or 2 total, which will use the same cond. and uncond. prompts for all.
            loss_rescale: To prevent fp16 underflow
        """

        if output_folder:
            output_folder.mkdir(parents=True, exist_ok=True)

        batch_size, _, height, width = image.shape

        if starting_lower_bound is not None:
            min_step = int(self.num_train_timesteps * starting_lower_bound)
            max_step = int(self.num_train_timesteps * starting_upper_bound)
            # select t, set multi-step diffusion
            T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
            self.scheduler.config.num_train_timesteps = T.item()
        else:
            self.scheduler.config.num_train_timesteps = self.num_train_timesteps

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        model_input = self.get_model_input(
            image=image,
            mask=mask,
            generator=generator,
            # self.scheduler.config.num_train_timesteps == 1000 is equivalent to starting_lower_bound and starting_upper_bound both being 1
            # so start with full noise by setting this to None
            starting_image=starting_image if self.scheduler.config.num_train_timesteps != 1000 else None,
            starting_timestep=self.scheduler.timesteps[0],
        )

        if depth is None:
            depth = torch.zeros_like(mask)

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )
        task1 = progress.add_task(
            f"[green]Inpainting batch of images...",
            total=len(self.scheduler.timesteps),
        )

        with progress:
            for i, t in enumerate(self.scheduler.timesteps):
                start_time = time.time()

                # DragDiffusion style guidance ("drag")
                use_drag_guidance = (
                    multiview_guidance_scale != 0.0 or reconstruction_guidance_scale != 0.0
                ) and i in guidance_steps
                if use_drag_guidance:
                    model_input.latents = model_input.latents.to(torch.float32).detach().requires_grad_(True)
                    scaler = torch.cuda.amp.GradScaler()
                    optimizer = torch.optim.Adam([model_input.latents], lr=1e-2)
                    for guidance_step in range(num_guidance_steps):
                        _, pred_original_sample, _ = self.get_noise_pred(
                            t,
                            model_input,
                            text_embeddings,
                            text_guidance_scale=text_guidance_scale,
                            image_guidance_scale=image_guidance_scale,
                            denoise_in_grid=denoise_in_grid,
                            multidiffusion_steps=1,
                            multidiffusion_type=multidiffusion_type,
                            randomize_latents=randomize_latents,
                            randomize_within_grid=randomize_within_grid,
                        )
                        x0 = self.decode_latents(
                            pred_original_sample,
                            use_decoder_approximation=use_decoder_approximation,
                        ).to(torch.float32)
                        x0 = torch.where(mask == 0, image, x0) if replace_original_pixels else x0

                        if output_folder:
                            image_x0 = torch.cat(list(x0.permute(0, 2, 3, 1)), dim=1).detach().cpu()
                            mediapy.write_image(
                                output_folder / f"x0-{i:06d}-{guidance_step:06d}.png",
                                image_x0,
                            )

                        loss = self.get_loss(
                            x0=x0,
                            mask=mask,
                            depth=depth,
                            multiview_guidance_scale=multiview_guidance_scale,
                            reconstruction_guidance_scale=reconstruction_guidance_scale,
                            feature_extractor=feature_extractor,
                            multiview_metric=multiview_metric,
                            K=K,
                            c2w=c2w,
                            output_folder=output_folder / "drag_guidance",
                            step=i,
                            guidance_step=guidance_step,
                            starting_image=starting_image,
                        )
                        if wandb.run is not None:
                            wandb.log({f"{output_folder.name}/drag_guidance_loss-{i}": loss})

                        optimizer.zero_grad()
                        assert not loss.isnan().any()
                        scaler.scale(loss).backward()

                        assert not model_input.latents.grad.isnan().any()
                        # print(
                        #     model_input.latents.grad.abs().mean(),
                        #     (model_input.latents.grad == 0.0).sum() / model_input.latents.grad.numel(),
                        # )

                        scaler.step(optimizer)
                        assert not model_input.latents.isnan().any()
                        assert not depth.isnan().any()
                        scaler.update()

                # take a step
                use_classifier_guidance = classifier_guidance_scale != 0.0 and i >= classifier_guidance_start_step
                model_input.latents = (
                    model_input.latents.to(self.dtype).detach().requires_grad_(use_classifier_guidance)
                )
                with torch.enable_grad() if use_classifier_guidance else torch.no_grad():
                    _, pred_original_sample, noise_pred = self.get_noise_pred(
                        t,
                        model_input,
                        text_embeddings,
                        text_guidance_scale=text_guidance_scale,
                        image_guidance_scale=image_guidance_scale,
                        denoise_in_grid=denoise_in_grid,
                        multidiffusion_steps=multidiffusion_steps,
                        multidiffusion_type=multidiffusion_type,
                        randomize_latents=randomize_latents,
                        randomize_within_grid=randomize_within_grid,
                    )

                    # classifier guidance ("classifier")
                    if use_classifier_guidance:
                        x0 = self.decode_latents(
                            pred_original_sample,
                            use_decoder_approximation=use_decoder_approximation,
                        ).to(torch.float32)
                        x0 = torch.where(mask == 0, image, x0) if replace_original_pixels else x0

                        loss = self.get_loss(
                            x0=x0,
                            mask=mask,
                            depth=depth,
                            multiview_guidance_scale=multiview_guidance_scale,
                            reconstruction_guidance_scale=reconstruction_guidance_scale,
                            feature_extractor=feature_extractor,
                            multiview_metric=multiview_metric,
                            K=K,
                            c2w=c2w,
                            output_folder=output_folder / "classifier_guidance",
                            step=i,
                            guidance_step=0,
                            starting_image=starting_image,
                        )
                        if wandb.run is not None:
                            wandb.log({f"{output_folder.name}/classifier_guidance_loss": loss})

                        grad = (
                            torch.autograd.grad(
                                classifier_guidance_loss_rescale * loss,
                                model_input.latents,
                            )[0]
                            / classifier_guidance_loss_rescale
                        )
                        # print(
                        #     grad.abs().mean(),
                        #     (grad == 0.0).sum() / grad.numel(),
                        # )
                        noise_pred = noise_pred + classifier_guidance_scale * grad

                    model_input.latents = model_input.latents.detach().requires_grad_(False)
                    scheduler_output = self.scheduler.step(noise_pred, t, model_input.latents, generator=generator)
                    model_input.latents = scheduler_output.prev_sample

                if output_folder:
                    # save the denoised x0
                    with torch.no_grad():
                        x0 = self.decode_latents(
                            pred_original_sample,
                            use_decoder_approximation=use_decoder_approximation,
                        ).to(torch.float32)
                        x0 = torch.where(mask == 0, image, x0) if replace_original_pixels else x0

                        if use_drag_guidance or use_classifier_guidance:
                            loss = self.get_loss(
                                x0=x0,
                                mask=mask,
                                depth=depth,
                                multiview_guidance_scale=multiview_guidance_scale,
                                reconstruction_guidance_scale=reconstruction_guidance_scale,
                                feature_extractor=feature_extractor,
                                multiview_metric=multiview_metric,
                                K=K,
                                c2w=c2w,
                                output_folder=None,
                                step=i,
                                guidance_step=0,
                                starting_image=starting_image,
                            )
                            if wandb.run is not None:
                                wandb.log({f"{output_folder.name}/loss": loss})

                    image_x0 = torch.cat(list(x0.permute(0, 2, 3, 1)), dim=1).detach().cpu()
                    mediapy.write_image(output_folder / "x0.png", image_x0)
                    mediapy.write_image(output_folder / f"x0-{i:06d}.png", image_x0)

                progress.update(task1, advance=1)
                end_time = time.time()
                # print(f"[green]Time for iter {i}:", end_time - start_time)

        if output_folder:
            output_filename = str(output_folder) + ".mp4"
            CONSOLE.print(f"[green]Saving video to {output_filename}")
            save_video_from_path(
                path=output_folder,
                glob_str="x0*png",
                sec=10,
                output_filename=output_filename,
            )

        with torch.no_grad():
            x0 = self.decode_latents(
                model_input.latents.detach(),
                use_decoder_approximation=use_decoder_approximation,
            ).to(torch.float32)
            x0 = torch.where(mask == 0, image, x0) if replace_original_pixels else x0
        return x0

    def encode_images(self, imgs: Float[Tensor, "B 3 512 512"]) -> Float[Tensor, "B 4 64 64"]:
        imgs = imgs * 2.0 - 1.0
        sampled_posterior = self.vae.encode(imgs.to(self.vae_device), return_dict=False)[0].sample().to(self.device)
        latents = sampled_posterior * 0.18215
        return latents

    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        use_decoder_approximation: bool = False,
    ) -> Float[Tensor, "B 3 Hout Wout"]:
        if use_decoder_approximation:
            da = get_decoder_approximation().to(latents)
            x = torch.nn.functional.interpolate(latents, scale_factor=self.vae_scale_factor, mode="bilinear")
            x = torch.matmul(x.permute(0, 2, 3, 1), da).permute(0, 3, 1, 2)
            return x
        else:
            scaled_latents = 1 / 0.18215 * latents
            image = self.vae.decode(scaled_latents.to(self.vae_device), return_dict=False)[0].to(self.device)
            image = (image * 0.5 + 0.5).clamp(0, 1)
            return image

    def sds_loss(
        self,
        text_embeddings: Union[Float[Tensor, "BB 77 768"], Float[Tensor, "2 77 768"]],
        image: Float[Tensor, "B 3 H W"],
        mask: Float[Tensor, "B 1 H W"],
        starting_image: Float[Tensor, "B 3 H W"],
        text_guidance_scale: Optional[float] = None,
        image_guidance_scale: Optional[float] = None,
        starting_lower_bound: float = 0.02,
        starting_upper_bound: float = 0.98,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> torch.Tensor:
        """Score Distilation Sampling loss proposed in DreamFusion paper (https://dreamfusion3d.github.io/)
        Args:
            text_embeddings: Text embeddings
            image: Rendered image
            mask: Mask, inpaint where 1
            text_guidance_scale: How much to weigh the guidance
            image_guidance_scale: How much to weigh the guidance
        Returns:
            The loss
        """

        # NOTE: doesn't work for gridding right now

        batch_size, _, height, width = image.shape

        min_step = int(self.num_train_timesteps * starting_lower_bound)
        max_step = int(self.num_train_timesteps * starting_upper_bound)

        t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)

        model_input = self.get_model_input(
            image=image,
            mask=mask,
            generator=generator,
            starting_image=starting_image,
            starting_timestep=t,
            keep_grad=True,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            _, _, noise_pred = self.get_noise_pred(
                t,
                model_input,
                text_embeddings,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                only_noise_pred=True,
            )

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]

        grad = w * (noise_pred - model_input.noise)
        grad = torch.nan_to_num(grad)

        target = (model_input.latents - grad).detach()
        loss = (
            0.5
            * torch.nn.functional.mse_loss(model_input.latents, target, reduction="sum")
            / model_input.latents.shape[0]
        )

        return loss


class RGBInpainterXL(RGBInpainter):
    def set_pipe(self):
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.dtype,
        }
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            **pipe_kwargs,
        )

    def setup(self):
        # Load LoRA
        if self.lora_model_path:
            self.pipe.load_lora_weights(self.lora_model_path)
            print(f"Loaded LoRA model from {self.lora_model_path}")

        # self.tokenizer = self.pipe.tokenizer
        # self.text_encoder = self.pipe.text_encoder.to(self.device).eval()
        self.pipe.to(self.device)

        self.unet = self.pipe.unet.to(self.device).eval()
        self.vae = self.pipe.vae.to(self.vae_device).eval()

        self.vae_scale_factor = 2 ** (len(self.pipe.vae.config.block_out_channels) - 1)
        self.vae_latent_channels = self.pipe.vae.config.latent_channels

        # self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.num_train_timesteps = self.scheduler.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        # save this in order to delete the pipeline after text encoding
        self.text_encoder_2_config_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        print(f"Loaded RGB inpainter!")

    def compute_text_embeddings(self, prompt: str, negative_prompt: str):
        """Get the text embeddings for a string."""
        assert self.pipe is not None

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(prompt, negative_prompt, device=self.device)
        return [
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ]

    def remove_pipe(self):
        del self.pipe
        cleanup()

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        requires_aesthetics_score=False,
    ):
        if requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.text_encoder_2_config_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    def forward_unet(
        self,
        sample,
        t,
        text_embeddings,
        denoise_in_grid: bool = False,
    ):
        # process embeddings
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = text_embeddings

        batch_size = sample.shape[0] // 3

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        height, width = sample.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = (height, width)
        target_size = (height, width)

        crops_coords_top_left = (0, 0)
        aesthetic_score = 6.0
        negative_aesthetic_score = 2.5
        negative_crops_coords_top_left = (0, 0)

        negative_original_size = original_size
        negative_target_size = target_size

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=self.dtype,
        )

        prompt_embeds = torch.cat(
            [
                prompt_embeds.repeat(batch_size, 1, 1),
                negative_prompt_embeds.repeat(batch_size, 1, 1),
                negative_prompt_embeds.repeat(batch_size, 1, 1),
            ],
            dim=0,
        )
        add_text_embeds = torch.cat(
            [
                pooled_prompt_embeds.repeat(batch_size, 1),
                negative_pooled_prompt_embeds.repeat(batch_size, 1),
                negative_pooled_prompt_embeds.repeat(batch_size, 1),
            ],
            dim=0,
        )
        add_time_ids = torch.cat(
            [
                add_time_ids.repeat(batch_size, 1),
                add_neg_time_ids.repeat(batch_size, 1),
                add_neg_time_ids.repeat(batch_size, 1),
            ],
            dim=0,
        )

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device)

        if denoise_in_grid:
            grid_sample = make_grid(sample)
            grid_prompt_embeds = prompt_embeds[:3].repeat(grid_sample.shape[0] // 3, 1, 1)
            grid_add_text_embeds = add_text_embeds[:3].repeat(grid_sample.shape[0] // 3, 1)
            grid_add_time_ids = add_time_ids[:3].repeat(grid_sample.shape[0] // 3, 1)
            added_cond_kwargs = {
                "text_embeds": grid_add_text_embeds,
                "time_ids": grid_add_time_ids,
            }
            noise_pred = self.unet(
                sample=grid_sample,
                timestep=t,
                encoder_hidden_states=grid_prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            noise_pred = undo_grid(noise_pred)
        else:
            added_cond_kwargs = {
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            }
            noise_pred = self.unet(
                sample=sample,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        return noise_pred