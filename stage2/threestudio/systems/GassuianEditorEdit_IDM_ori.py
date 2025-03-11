
from dataclasses import dataclass, field
import random
import shutil

from tqdm import tqdm

import torch
import threestudio
import os
from diffusers.utils import load_image
from threestudio.utils.clip_metrics import ClipSimilarity





from diffusers import AutoPipelineForInpainting, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL, StableDiffusionInpaintPipeline, DDIMScheduler
from threestudio.systems.GaussianEditor_IDM_ori import GaussianEditor_Video_Copy
from torchvision.io import read_image

from gaussiansplatting.utils.loss_utils import ssim
import time
from torchvision.utils import save_image
import cv2
from threestudio.systems.inference_HumanParsing.Human_Parsing.utils.inference_funcs import inference_parsing
from threestudio.systems.inference_HumanParsing.Human_Parsing.inference import HumanParsing
from PIL import Image
import numpy as np
from diffusers.image_processor import VaeImageProcessor
import torch.nn.functional as F
from torchvision.utils import save_image

from torchvision.transforms.functional import pil_to_tensor, to_pil_image

        
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid

@threestudio.register("gsedit-system-edit-idm-ori")
class GaussianEditor_Edit_Video_Copy(GaussianEditor_Video_Copy):
    @dataclass
    class Config(GaussianEditor_Video_Copy.Config):
        local_edit: bool = False

        seg_prompt: str = ""

        second_guidance_type: str = "dds"
        second_guidance: dict = field(default_factory=dict)
        dds_target_prompt_processor: dict = field(default_factory=dict)
        dds_source_prompt_processor: dict = field(default_factory=dict)

        clip_prompt_origin: str = ""
        clip_prompt_target: str = ""  # only for metrics

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join(
                "edit_cache_Copy2", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join(
                "edit_cache_Copy2", self.cfg.gs_source.replace("/", "-"))

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render_Copy")


        self.update_mask_viton(body_part='upper', replace=True)
        self.render_all_view_Inpaint(
            category='up', cache_name="Inpaint_Render_hoodie_Batch_Up")
        

        self.update_mask_viton(body_part='both', replace=False)

        self.rendered_edited_img = self.edited_frames_Inpaint_up
        

        


    def training_step(self, batch, batch_idx):
        self.gaussian.update_learning_rate(self.true_global_step)
        

        
        if  (self.true_global_step % self.cfg.renew_render_interval ==0) and self.cfg.enable_reInpaint==True and self.true_global_step!=0 and self.true_global_step>self.cfg.which_step:
            torch.cuda.empty_cache()
            self.cfg.cache_overwrite=True
            self.render_all_view("reRender_ReInpaint")
            self.rndndn=None
            self.batch_size_pipe=2
            self.render_all_view_Inpaint(category='reinpaint', cache_name="Inpaint_Render_reinpaint",flag=True)
            


        batch_index = batch["index"]
        if isinstance(batch_index, int):
            batch_index = [batch_index]

        
        out = self(batch, local=False)
        images = out["comp_rgb"]

        loss = 0.0

        
        edited_img = self.rendered_edited_img[batch_index[0]].unsqueeze(
                0)/255.0        
        
                   

        edited_img = F.interpolate(
            edited_img, (self.global_height, self.global_width), mode="bilinear").float().cuda()
        
        
        
        guidance_out = {
            "loss_l1": 1.5*torch.nn.functional.l1_loss(images.permute(0, 3, 1, 2), edited_img),
            "loss_p": self.perceptual_loss(images.permute(0, 3, 1, 2).contiguous(), edited_img.contiguous(),).sum(),
        }
        
        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        if (
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        return {"loss": loss}



    def on_validation_epoch_end(self):
        if len(self.cfg.clip_prompt_target) > 0:
            self.compute_clip()

    def compute_clip(self):
        clip_metrics = ClipSimilarity().to(self.gaussian.get_xyz.device)
        total_cos = 0
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                cur_batch = {
                    "index": id,
                    "camera": [cur_cam],
                    "height": self.trainer.datamodule.train_dataset.height,
                    "width": self.trainer.datamodule.train_dataset.width,
                }
                out = self(cur_batch)["comp_rgb"]
                _, _, cos_sim, _ = clip_metrics(self.origin_frames[id].permute(0, 3, 1, 2), out.permute(0, 3, 1, 2),
                                                self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target)
                total_cos += abs(cos_sim.item())
        print(self.cfg.clip_prompt_origin, self.cfg.clip_prompt_target,
              total_cos / len(self.view_list))
        self.log("train/clip_sim", total_cos / len(self.view_list))
