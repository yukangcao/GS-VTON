
from dataclasses import dataclass, field
import random
import re

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
import threestudio
import os
from threestudio.systems.base import BaseLift3DSystem
from torchvision.io import read_image
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
import torchvision
from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from diffusers.utils import load_image
from omegaconf import OmegaConf
from rembg import remove,new_session
from argparse import ArgumentParser
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
# from threestudio.utils.sam import LangSAMTextSegmentor
import time
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)

from diffusers import ControlNetModel, UniPCMultistepScheduler,StableDiffusionControlNetInpaintPipeline
import re
import glob


from threestudio.preprocess.humanparsing.run_parsing import Parsing
from threestudio.preprocess.openpose.run_openpose import OpenPose
from threestudio.utils_mask import get_mask_location
import sys


from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel

from torchvision.transforms.functional import pil_to_tensor,to_pil_image
from torchvision.utils import save_image

session = new_session()


from diffusers.utils import USE_PEFT_BACKEND
import xformers.ops as xops
from einops import rearrange
import torch
import torch.nn.functional as F

from  diffusers.utils.torch_utils  import randn_tensor

from threestudio.systems.attention_utils import *

def extract_cloth(image,parsing_model):
    model_parse, _ = parsing_model(image)
    parse_array = np.array(model_parse)
    cloth_mask=parse_array==4
    cloth_img=np.array(image) * np.repeat(cloth_mask[...,None],3,axis=2)
    parse_array=Image.fromarray(cloth_img)
    return parse_array

def extract_number_from_path(path):
    match = re.search(r'/(\d{4})/', path)
    return match.group(1) if match else None

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask



class GaussianEditor_Video_Copy(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        gs_source: str = None

        per_editing_step: int = -1
        edit_begin_step: int = 0
        edit_until_step: int = 4000

        densify_until_iter: int = 4000
        densify_from_iter: int = 0
        densification_interval: int = 100
        max_densify_percent: float = 0.01

        gs_lr_scaler: float = 1
        gs_final_lr_scaler: float = 1
        color_lr_scaler: float = 1
        opacity_lr_scaler: float = 1
        scaling_lr_scaler: float = 1
        rotation_lr_scaler: float = 1

        # lr
        mask_thres: float = 0.5
        max_grad: float = 1e-7
        min_opacity: float = 0.005

        seg_prompt: str = ""

        # cache
        cache_overwrite: bool = True
        
        cache_dir: str = ""

        # anchor
        anchor_weight_init: float = 0.1
        anchor_weight_init_g0: float = 1.0
        anchor_weight_multiplier: float = 2
        training_args: dict = field(default_factory=dict)
        
        enable_attention: bool = True
        enable_ControlNet: bool = False
        inpaint_prompt:str="a photo of sks"
        control_image_path:str=""
        inpaint_model_path:str=""
        which_step:int=0
        renew_render_interval:int=0
        use_gan: bool = False
        enable_reInpaint: bool = True
        enable_limit: bool= True
        
        second_phase: bool = True
        
        global_height:int=0    
        global_width:int=0        
        
        ref1_num:int=0                
        ref2_num:int=0                
        ref3_num:int=0       
        

    cfg: Config

    def configure(self) -> None:
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=self.cfg.anchor_weight_init_g0,
            anchor_weight_init=self.cfg.anchor_weight_init,
            anchor_weight_multiplier=self.cfg.anchor_weight_multiplier,
        )
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self.text_segmentor = None
        
        
        self.openpose_model = OpenPose(0)
        self.openpose_model.preprocessor.body_estimation.model.to('cuda')
        self.parsing_model = Parsing(0)
        
        self.inference_Viton=None
        

        self.mask_dict_up = {}
        self.mask_dict_down = {}
        self.mask_dict = {}
        self.mask_dict_up_tensor = {}
        self.mask_dict_down_tensor = {}     
        self.pipe_Inpaint=None   
        self.rndndn=None

        
        self.edited_frames_Inpaint_up={}
        self.edited_frames_Inpaint_down={}    
        self.reinpaint_frames={}       
        self.saved_batch_ids=None   

        self.global_height=self.cfg.global_height
        self.global_width=self.cfg.global_width
        
        self.batch_size_pipe=8

        
        


    def create_directories(self, replace):
        base_path = './threestudio/systems/temp_data_tobe_used'
        if replace:
            dirs = [
                f'{base_path}/mask_openpose_combine_rembg',
                f'{base_path}/mask_openpose_combine_rembg_Tensor_Up',
                f'{base_path}/mask_openpose_combine_rembg_Tensor_Down'
            ]
            for dir_path in dirs:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
        
        os.makedirs(f'{base_path}/mask_openpose_combine_rembg', exist_ok=True)
        os.makedirs(f'{base_path}/mask_openpose_combine_rembg_Tensor_Up', exist_ok=True)
        os.makedirs(f'{base_path}/mask_openpose_combine_rembg_Tensor_Down', exist_ok=True)

    def generate_mask(self, id, body_part):
        if body_part == "upper":
            mask, mask_pil = self.mask_genration_rembg_Infer_Basic(self.origin_frames[id], "upper_body")
        else:
            mask, mask_pil = self.mask_genration_rembg_Infer(self.edited_frames_Inpaint_up[id].permute(1,2,0).unsqueeze(0).float()/255.0, "lower_body")
        return mask, mask_pil
    @torch.no_grad()
    def update_mask_viton_upper(self):
        for id in tqdm(self.view_list):
            path_up = f'./threestudio/systems/temp_data_tobe_used/mask_openpose_combine_rembg_Tensor_Up/{id}.pt'
            if not os.path.exists(path_up):
                mask_up, mask_pil_up = self.generate_mask(id, "upper")
                self.mask_dict_up[id] = mask_pil_up
                self.mask_dict_up_tensor[id] = mask_up
                torch.save(mask_up, path_up)
                mask_pil_up.save(path_up.replace('.pt', '.png'))
            else:
                self.mask_dict_up_tensor[id] = torch.load(path_up)
                self.mask_dict_up[id] = Image.open(path_up.replace('.pt', '.png'))

    def update_mask_viton_lower(self):
        for id in tqdm(self.view_list):
            path_down = f'./threestudio/systems/temp_data_tobe_used/mask_openpose_combine_rembg_Tensor_Down/{id}.pt'
            if not os.path.exists(path_down):
                mask_down, mask_pil_down = self.generate_mask(id, "lower")
                self.mask_dict_down[id] = mask_pil_down
                self.mask_dict_down_tensor[id] = mask_down
                torch.save(mask_down, path_down)
                mask_pil_down.save(path_down.replace('.pt', '.png'))
            else:
                self.mask_dict_down_tensor[id] = torch.load(path_down)
                self.mask_dict_down[id] = Image.open(path_down.replace('.pt', '.png'))
    @torch.no_grad()
    def update_mask_viton(self, body_part, replace=True):
        

        self.create_directories(replace)
        
        if body_part == "upper":
            self.update_mask_viton_upper()
        elif body_part == "lower":
            self.update_mask_viton_lower()
        else:
            weights = torch.zeros_like(self.gaussian._opacity)
            weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
            for id in tqdm(self.view_list):
                path1 = f'./threestudio/systems/temp_data_tobe_used/mask_openpose_combine_rembg/{id}.pt'
                if not os.path.exists(path1):
                    mask_up = self.mask_dict_up_tensor.get(id)
                    mask = mask_up
                    self.mask_dict[id] = mask
                    try:
                        torch.save(mask, path1)
                    except:
                        pass
                    to_pil_image(mask).save(path1.replace('.pt', '.png'))
                else:
                    self.mask_dict[id] = torch.load(path1)

                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, self.mask_dict[id])
            
            weights /= weights_cnt + 1e-7
            selected_mask = weights > self.cfg.mask_thres
            selected_mask = selected_mask[:, 0]
            self.gaussian.set_mask(selected_mask)
            self.gaussian.apply_grad_mask(selected_mask)
            self.mask = selected_mask


    def CATVTON_Mask(self,input_img):
        save_image(input_img[0].permute(2,0,1),'./temp_image_to_get_mask.png')
        input_img=load_image('./temp_image_to_get_mask.png').resize((self.global_width,self.global_height))
        
        masks = self.automasker(input_img, 'upper')['mask']
        mask_1 = self.mask_processor.blur(masks, blur_factor=9)
        mask_1_tensor=pil_to_tensor(mask_1).cuda() * 1.0
        return mask_1_tensor,mask_1
        

    def mask_genration_rembg_Infer(self,input_img,category="upper_body",intersect=False):
        save_image(input_img[0].permute(2,0,1),'./temp_image_to_get_mask.png')
        with torch.no_grad():
            human_img=load_image('./temp_image_to_get_mask.png').resize((self.global_width,self.global_height))
            
            model_parse, _ = self.parsing_model(human_img.resize((384,512)))
            
            parse_array=np.array(model_parse)
                    
            parse_head = (parse_array == 1).astype(np.float32) + \
                        (parse_array == 3).astype(np.float32) + \
                        (parse_array == 11).astype(np.float32)            
            
            parse_mask = ((parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32) +  (parse_array == 15).astype(np.float32) + (parse_array == 14)).astype(np.float32)
            parse_mask = cv2.dilate(parse_mask, np.ones((3, 3), np.uint16), iterations=5)
            neck_mask = (parse_array == 18).astype(np.float32)
            neck_mask = cv2.dilate(neck_mask, np.ones((3, 3), np.uint16), iterations=1)
            neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
            parse_mask = np.logical_or(parse_mask, neck_mask)
            
            dst = hole_fill(parse_mask.astype(np.uint8))
            dst = refine_mask(dst)
            inpaint_mask = dst * 1
            
            mask_1_down=Image.fromarray(inpaint_mask)
            
            mask_1_down = mask_1_down.resize(human_img.size)    
            
            mask_1=mask_1_down
            
            mask_1_tensor=pil_to_tensor(mask_1).cuda() * 1.0


            return mask_1_tensor,mask_1
        
    @torch.no_grad()        
    def mask_genration_rembg_Infer_Basic(self,input_img,category="upper_body",intersect=False):
        save_image(input_img[0].permute(2,0,1),'./temp_image_to_get_mask.png')
        with torch.no_grad():
            human_img=load_image('./temp_image_to_get_mask.png').resize((self.global_width,self.global_height))
            
            keypoints = self.openpose_model(human_img.resize((384,512)))
            model_parse, _ = self.parsing_model(human_img.resize((384,512)))
            mask_1_down, _ = get_mask_location('hd', category, model_parse, keypoints)
            
            
            mask_1_down = mask_1_down.resize(human_img.size)    
            mask_1_down=pil_to_tensor(mask_1_down).cuda()
            mask_1_down=(mask_1_down/255.0).bool()
            
                
            mask_1=mask_1_down.float()
            if intersect==True:
                rembg_mask= remove(human_img,only_mask=True,session=session,post_process_mask=True)
                rembg_mask=((pil_to_tensor(rembg_mask).cuda())/255.0).bool()
                mask_1=torch.logical_and(mask_1,rembg_mask).float()
                
            mask_1=to_pil_image(mask_1)
            
            erode_kernel = ImageFilter.MaxFilter(3)
            mask_1 = mask_1.filter(erode_kernel)
            
            
            mask_1_tensor=pil_to_tensor(mask_1).cuda() * 1.0


            return mask_1_tensor,mask_1      
     

    @torch.no_grad()
    def update_mask_new(self) -> None:
        weights = torch.zeros_like(self.gaussian._opacity).unsqueeze(0).repeat(20,1,1)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32).unsqueeze(0).repeat(20,1,1)
        desired_label=5 #shirt    
        logit_tensors=torch.load('./Self_Correction_Human_Parsing/logits.pt')
        with torch.no_grad():
            for id in tqdm(self.view_list):
                if id==0:
                    continue
                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                parsing_result = logit_tensors[id].cuda()
                for labels_idx in range(20):
                    mask=(parsing_result==labels_idx).float().unsqueeze(0)
                    self.gaussian.apply_weights(cur_cam, weights[labels_idx], weights_cnt[labels_idx], mask)
        
        weights /= weights_cnt + 1e-7
        selected_mask = weights > self.cfg.mask_thres
        selected_mask = selected_mask[:,:, 0]
        self.gaussian.set_mask(selected_mask[desired_label])
        self.gaussian.apply_grad_mask(selected_mask[desired_label])
        self.mask=selected_mask[desired_label]
        




    @torch.no_grad()
    def update_mask(self, save_name="mask") -> None:
        print(f"Segment with prompt: {self.cfg.seg_prompt}")
        mask_cache_dir = os.path.join(
            self.cache_dir, self.cfg.seg_prompt + f"_{save_name}_{self.view_num}_view"
        )
        gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")
        if not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
            if os.path.exists(mask_cache_dir):
                shutil.rmtree(mask_cache_dir)
            os.makedirs(mask_cache_dir)
            weights = torch.zeros_like(self.gaussian._opacity)
            weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
            threestudio.info(f"Segmentation with prompt: {self.cfg.seg_prompt}")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_path_viz = os.path.join(
                    mask_cache_dir, "viz_{:0>4d}.png".format(id)
                )

                cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]

                mask = self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt)[
                    0
                ].to(get_device())

                mask_to_save = (
                        mask[0]
                        .cpu()
                        .detach()[..., None]
                        .repeat(1, 1, 3)
                        .numpy()
                        .clip(0.0, 1.0)
                        * 255.0
                ).astype(np.uint8)
                cv2.imwrite(cur_path, mask_to_save)

                masked_image = self.origin_frames[id].detach().clone()[0]
                masked_image[mask[0].bool()] *= 0.3
                masked_image_to_save = (
                        masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                ).astype(np.uint8)
                masked_image_to_save = cv2.cvtColor(
                    masked_image_to_save, cv2.COLOR_RGB2BGR
                )
                cv2.imwrite(cur_path_viz, masked_image_to_save)
                self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

            weights /= weights_cnt + 1e-7

            selected_mask = weights > self.cfg.mask_thres
            selected_mask = selected_mask[:, 0]
            torch.save(selected_mask, gs_mask_path)
        else:
            print("load cache")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_mask = cv2.imread(cur_path)
                cur_mask = torch.tensor(
                    cur_mask / 255, device="cuda", dtype=torch.float32
                )[..., 0][None]
            selected_mask = torch.load(gs_mask_path)

        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)

    def on_validation_epoch_end(self):
        pass
    


    def forward(self, batch: Dict[str, Any], renderbackground=None, local=False,input_gaussian=None) -> Dict[str, Any]:
        if input_gaussian==None:
            input_gaussian=self.gaussian
        
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        semantics = []
        masks = []
        self.viewspace_point_list = []
        input_gaussian.localize = local

        for id, cam in enumerate(batch["camera"]):
            render_pkg = render(cam, input_gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)

            semantic_map = render(
                cam,
                input_gaussian,
                self.pipe,
                renderbackground,
                override_color=input_gaussian.mask[..., None].float().repeat(1, 3),
            )["render"]
            semantic_map = torch.norm(semantic_map, dim=0)
            semantic_map = semantic_map > 0.8
            semantic_map_viz = image.detach().clone()
            semantic_map_viz = semantic_map_viz.permute(
                1, 2, 0
            )  # 3 512 512 to 512 512 3
            semantic_map_viz[semantic_map] = 0.40 * semantic_map_viz[
                semantic_map
            ] + 0.60 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
            semantic_map_viz = semantic_map_viz.permute(
                2, 0, 1
            )  # 512 512 3 to 3 512 512

            semantics.append(semantic_map_viz)
            masks.append(semantic_map)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        input_gaussian.localize = False  # reverse

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        semantics = torch.stack(semantics, dim=0)
        masks = torch.stack(masks, dim=0)

        render_pkg["semantic"] = semantics
        render_pkg["masks"] = masks
        self.visibility_filter = self.radii > 0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def render_all_view(self, cache_name):
        
        
        cache_dir = os.path.join(self.cache_dir, cache_name)
        if os.path.exists(cache_dir) and self.cfg.cache_overwrite:
            shutil.rmtree(cache_dir)
                
        os.makedirs(cache_dir, exist_ok=True)
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    out = self(cur_batch)["comp_rgb"]
                    # print('out:',out.shape)
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)
                cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                self.origin_frames[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]
                    
    @torch.no_grad()
    def utils_1111(self,category='up'):
        if category=='up' or category=='reinpaint':
                if self.cfg.enable_ControlNet:
                    controlnet_path = "shgao/edit-anything-v0-3"
                    # controlnet_path = "/root/autodl-tmp/cache/hub/models--shgao--edit-anything-v0-3/snapshots/14a35cb4e0dd574a6e895b7f73c827acadc0970f"
                    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32) 
                    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-inpainting", controlnet=controlnet, torch_dtype=torch.float32
                        ,vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
                    )            
                else:
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-2-inpainting",
                        torch_dtype=torch.float32,
                        revision=None,
                        vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse'),
                        safety_checker=None,
                    )

                pipe.unet = UNet2DConditionModel.from_pretrained(
                    self.cfg.inpaint_model_path, subfolder="unet", revision=None,
                )
                pipe.text_encoder = CLIPTextModel.from_pretrained(
                    self.cfg.inpaint_model_path, subfolder="text_encoder", revision=None,
                )
                pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
                pipe = pipe.to("cuda")
                
                
                if self.cfg.enable_attention:
                    if category=='reinpaint':
                        self.attn_prcs=XFormersAttnProcessor_Reference_BetterImpl_Multi_MoreThanTwo(ref_num=self.num_refFrame,enable_limit=self.cfg.enable_limit)
                        pipe.unet.set_attn_processor(processor=self.attn_prcs)    
                    elif category=='up':
                        
                        self.attn_prcs=XFormersAttnProcessor_Reference_BetterImpl_Multi_MoreThanTwo(ref_num=self.num_refFrame,enable_limit=self.cfg.enable_limit)
                        
                        pipe.unet.set_attn_processor(processor=self.attn_prcs)    
                    else:
                        pipe.enable_xformers_memory_efficient_attention()
                else:
                    pipe.enable_xformers_memory_efficient_attention()
                return pipe
            
            
        elif category=='down':
            exit(1)
            # pipe_2 = StableDiffusionInpaintPipeline.from_pretrained(
            #     "stabilityai/stable-diffusion-2-inpainting",
            #     torch_dtype=torch.float32,
            #     revision=None,
            #     vae=AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse'),
            # )

            # pipe_2.unet = UNet2DConditionModel.from_pretrained(
            #     '/root/autodl-tmp/Lora/realfill/trouser-model', subfolder="unet", revision=None,
            # )
            
            # pipe_2.text_encoder = CLIPTextModel.from_pretrained(
            #     '/root/autodl-tmp/Lora/realfill/trouser-model', subfolder="text_encoder", revision=None,
            # )
            # pipe_2.scheduler = DDPMScheduler.from_config(pipe_2.scheduler.config)
            # pipe_2 = pipe_2.to("cuda")   
            
            
            # pipe_2.unet.set_attn_processor(processor=XFormersAttnProcessor_Reference_BetterImpl_Multi_MoreThanTwo(ref_num=self.num_refFrame))    
            # return pipe_2


            
    @torch.no_grad()     
    def render_Inapint(self, images, mask_images, pipe,ctrl_scale):
        torch.cuda.empty_cache()
        w=8*round(self.global_width/8)
        h=8*round(self.global_height/8)                    
        generator = torch.Generator(device="cuda").manual_seed(0)
        if self.rndndn==None:
            self.rndndn=randn_tensor((1, pipe.vae.config.latent_channels, h // pipe.vae_scale_factor , w // pipe.vae_scale_factor), generator=generator, device=pipe._execution_device).repeat(len(images),1,1,1)
            

        if len(images)<self.rndndn.shape[0]:
            self.rndndn=self.rndndn[:len(images),:,:,:]
        
        
     
        
        if self.cfg.enable_ControlNet:
            results = pipe(
                [self.cfg.inpaint_prompt] * len(images),
                image=images,
                mask_image=mask_images,
                num_inference_steps=25, guidance_scale=1.0,
                negative_prompt=['monochrome, lowres, bad anatomy, worst quality, low quality'],
                height=h,
                width=w,           
                generator=generator,
                control_image=self.inpaint_img_controlnet,
                controlnet_conditioning_scale=ctrl_scale,
                latents = self.rndndn,
            ).images
        else:            
            results = pipe(
                [self.cfg.inpaint_prompt] * len(images),
                image=images,
                mask_image=mask_images,
                num_inference_steps=25, guidance_scale=1.0,
                negative_prompt=['monochrome, lowres, bad anatomy, worst quality, low quality'],
                height=h,
                width=w,    
                generator=generator,
                latents = self.rndndn,
            ).images            

        composite_results = []

        
        try:
            for result, image ,mask_image in zip(results, images,mask_images):
                composite_results.append(Image.composite(result.resize(image.size,Image.Resampling.LANCZOS), image, mask_image))
        except:
            composite_results.append(Image.composite(results[0].resize(len(images)), images, mask_images))
            
        return composite_results
    
    
    @torch.no_grad()         
    def render_all_view_Inpaint(self, category='up',cache_name="Inpaint_Render_hoodie_Batch",flag=False):
        torch.cuda.empty_cache()
        self.pipe_Inpaint=None
        # if self.cfg.cache_overwrite and self.pipe_Inpaint==None:
        if self.pipe_Inpaint==None:
            self.pipe_Inpaint = self.utils_1111(category)
            self.pipe_Inpaint.enable_model_cpu_offload()

        
        
        cache_dir = os.path.join(self.cache_dir, cache_name)
        
        if os.path.exists(cache_dir) and self.cfg.cache_overwrite:
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)           
        

               
               
        batch_size=self.batch_size_pipe
        
        with torch.no_grad():
            for id in tqdm(range(0, len(self.view_list), batch_size)):
                
                batch_id=id
                batch_ids = self.view_list[id:id+batch_size]

                              
                for index_id in self.ref_index:
                    batch_ids.insert(0,index_id)
                
                
                
                to_remove=[]
                            
                batch_images = []
                batch_masks = []
                for id in batch_ids:
                    torch.cuda.empty_cache()
                    cur_path = os.path.join(cache_dir, "{:0>4d}.pt".format(id))
                    if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                        if category=='up':
                            mask_image=self.mask_dict_up[id]
                            curr_frame = self.origin_frames[id]
                            batch_images.append(to_pil_image(curr_frame[0].permute(2, 0, 1)))
                        
                        elif category=='down':
                            mask_image=self.mask_dict_down[id]
                            curr_frame = torch.load(os.path.join(os.path.join(self.cache_dir, 'Inpaint_Render_hoodie_Batch_Up'), "{:0>4d}.pt".format(id)))
                            batch_images.append(to_pil_image(curr_frame))
                        
                        elif category=='reinpaint':
                            mask_image=self.mask_dict_up[id]
                            curr_frame = self.origin_frames[id]
                            batch_images.append(to_pil_image(curr_frame[0].permute(2, 0, 1)))
                            
                            
                        batch_masks.append((mask_image))
                    else:
                        if category=='up':
                            self.edited_frames_Inpaint_up[id] = torch.load(cur_path).cuda()
                        elif category=='down':
                            self.edited_frames_Inpaint_down[id] = torch.load(cur_path).cuda()
                        
                        if id not in self.ref_index:
                            to_remove.append(id)
                
                batch_ids=[fruit for fruit in batch_ids if fruit not in to_remove]
                
                if batch_images:
                    if self.pipe_Inpaint==None:
                        self.pipe_Inpaint = self.utils_1111(category)
                    
                    composite_results = self.render_Inapint(batch_images, batch_masks, self.pipe_Inpaint,ctrl_scale=0.05 )
                    tensor_images = [torchvision.transforms.ToTensor()(img) for img in composite_results]
                    torchvision.utils.save_image(tensor_images, f'{cache_dir}/grid{batch_id}_first.png')                    
                    
                    try:
                        self.attn_prcs.counter=0
                    except:
                        pass
                    
                    if self.cfg.second_phase==True:
                        composite_results = self.render_Inapint(composite_results, batch_masks, self.pipe_Inpaint,ctrl_scale=0.1)
                        tensor_images = [torchvision.transforms.ToTensor()(img) for img in composite_results]
                        torchvision.utils.save_image(tensor_images, f'{cache_dir}/grid{batch_id}_second.png')                    
                    
                    for idx, result in enumerate(composite_results):
                        result.save(os.path.join(cache_dir, "{:0>4d}.png".format(batch_ids[idx])))
                        img = pil_to_tensor(result)
                        
                        if category=='up':
                            self.edited_frames_Inpaint_up[batch_ids[idx]] = img
                        elif category=='down':
                            self.edited_frames_Inpaint_down[batch_ids[idx]] = img          
                        elif category=='reinpaint':
                            self.rendered_edited_img[batch_ids[idx]] = img
                            
                
            
        
    def on_before_optimizer_step(self, optimizer):
        
        
        with torch.no_grad():
            if self.true_global_step < self.cfg.densify_until_iter:
                viewspace_point_tensor_grad = torch.zeros_like(
                    self.viewspace_point_list[0]
                )
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = (
                            viewspace_point_tensor_grad
                            + self.viewspace_point_list[idx].grad
                    )
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter],
                    self.radii[self.visibility_filter],
                )
                self.gaussian.add_densification_stats(
                    viewspace_point_tensor_grad, self.visibility_filter
                )
                # Densification
                if (
                        self.true_global_step >= self.cfg.densify_from_iter
                        and self.true_global_step % self.cfg.densification_interval == 0
                ):  # 500 100
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.max_densify_percent,
                        self.cfg.min_opacity,
                        self.cameras_extent,
                        5,
                    )

    def validation_step(self, batch, batch_idx):
        batch["camera"] = [
            self.trainer.datamodule.train_dataset.scene.cameras[idx]
            for idx in batch["index"]
        ]
        out = self(batch)
        for idx in range(len(batch["index"])):
            cam_index = batch["index"][idx].item()
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][idx]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": self.origin_frames[cam_index][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.edit_frames[cam_index][0]
                            if cam_index in self.edit_frames
                            else torch.zeros_like(self.origin_frames[cam_index][0]),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name=f"validation_step_{idx}",
                step=self.true_global_step,
            )
            
            self.save_image_grid(
                f"render_it{self.true_global_step}-{batch['index'][idx]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["semantic"][idx].moveaxis(0, -1),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "semantic" in out
                    else []
                ),
                name=f"validation_step_render_{idx}",
                step=self.true_global_step,
            )
    

    def test_step(self, batch, batch_idx):
        only_rgb = True  # TODO add depth test step
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        batch["camera"] = [
            self.trainer.datamodule.val_dataset.scene.cameras[batch["index"]]
        ]
        testbackground_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
        save_list = []
        for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        if len(save_list) > 0:
            self.save_image_grid(
                f"edited_images.png",
                save_list,
                name="edited_images",
                step=self.true_global_step,
            )
        save_list = []
        for index, image in sorted(
                self.origin_frames.items(), key=lambda item: item[0]
        ):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        self.save_image_grid(
            f"origin_images.png",
            save_list,
            name="origin",
            step=self.true_global_step,
        )

        save_path = self.get_save_path(f"last.ply")
        self.inpaint_img_controlnet.save(self.get_save_path(f"input_cloth.png"))
        self.gaussian.save_ply(save_path)

    def configure_optimizers(self):

        # Read the numbers from the file
        
        with open('../stage1/selected_numbers.txt', 'r') as f:
            self.ref_index = [int(line.strip())-1 for line in f.readlines()]
            
        self.num_refFrame=len(self.ref_index)
        num1, num2, num3, num4 = self.ref_index
        print(f"\nAssigned numbers: {num1}, {num2}, {num3}, {num4}\n")

        
        
      
        
        self.parser = ArgumentParser(description="Training script parameters")
        self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
        
        for index in self.ref_index:
            self.view_list.insert(0,index)
        
        
           
        self.view_num = len(self.view_list)
        
        opt = OptimizationParams(self.parser, self.trainer.max_steps, self.cfg.gs_lr_scaler, self.cfg.gs_final_lr_scaler, self.cfg.color_lr_scaler,
                                 self.cfg.opacity_lr_scaler, self.cfg.scaling_lr_scaler, self.cfg.rotation_lr_scaler, )
        
        self.gaussian.load_ply(self.cfg.gs_source)
        
        
        
        
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.cameras_extent = self.trainer.datamodule.train_dataset.scene.cameras_extent
        self.gaussian.spatial_lr_scale = self.cameras_extent

        self.pipe = PipelineParams(self.parser)
        opt = OmegaConf.create(vars(opt))
        opt.update(self.cfg.training_args)
        self.gaussian.training_setup(opt)
        
        
 
        ret = {
            "optimizer": self.gaussian.optimizer,
            
        }
        

        
        
        
        
        print("enable_attention",self.cfg.enable_attention)
        print("enable_ControlNet",self.cfg.enable_ControlNet)
        print("inpaint_prompt",self.cfg.inpaint_prompt)
        print("control_image_path",self.cfg.control_image_path)
        print("inpaint_model_path",self.cfg.inpaint_model_path)
        
        
        self.global_height=self.cfg.global_height
        self.global_width=self.cfg.global_width      
        self.trainer.datamodule.train_dataset.height=self.global_height
        self.trainer.datamodule.train_dataset.width=self.global_width  
        self.inpaint_img_controlnet=load_image(self.cfg.control_image_path)        
        

        return ret
    