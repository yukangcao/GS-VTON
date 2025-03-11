import argparse
import random
import shutil
import sys
sys.path.append('./')
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List
from diffusers.utils import load_image
import torchvision
import torch
from basicsr.utils.download_util import load_file_from_url
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from modified_attn import XFormersAttnProcessor_Cat_Ref

print("\n Run IDM \n")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)



def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')


parsing_model = Parsing(0)
openpose_model = OpenPose(0)

def extract_cloth(image):
    model_parse, _ = parsing_model(image)
    parse_array = np.array(model_parse)
    cloth_mask=parse_array==4
    cloth_img=np.array(image) * np.repeat(cloth_mask[...,None],3,axis=2)
    parse_array=Image.fromarray(cloth_img)
    return parse_array

random.seed(0)


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--garm_img_path",
    type=str,
    
)
parser.add_argument(
    "--data_path_imgs",
    type=str,
)
parser.add_argument(
    "--height",
    type=int,
    default=512,
)
parser.add_argument(
    "--width",
    type=int,
    default=512,
)

parser.add_argument(
    "--garm_desc_given",
    type=str,
)
parser.add_argument(
    "--enable_idm_attn",
    type=lambda x: str(x).lower() == 'true',
)

args = parser.parse_args()


global_height = args.height
global_width =  args.width

global_height=8*round(global_height/8)
global_width=8*round(global_width/8)

garment_desc =  args.garm_desc_given

def load_model(base_path):
    unet = UNet2DConditionModel().from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
    unet.requires_grad_(False)
    tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
    tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
    noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

    text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
    vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,                        
)

    UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)
    

    UNet_Encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

    pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
    pipe.unet_encoder = UNet_Encoder
    
    return parsing_model,openpose_model,tensor_transform,pipe

_, _, tensor_transform, pipe = load_model(base_path)

def virtual_tryon(human_img_paths, garm_img_path, garment_desc, is_checked=True, is_checked_crop=False, denoise_steps=22, seed=42):
    openpose_model.preprocessor.body_estimation.model.to(device)

    print("garment_descX:",garment_desc)
    
    
    garm_img = load_image(garm_img_path).resize((global_width, global_height))
    
    human_img_batch=[]
    mask_batch=[]
    mask_gray_batch=[]
    pose_img_batch=[]
    
    
    

    for human_img_path in human_img_paths:
        human_img_orig = load_image(human_img_path)
        human_img_, mask_, mask_gray_, pose_img_ = batch_preparation(is_checked, is_checked_crop, human_img_orig)
        human_img_batch.append(human_img_)
        mask_batch.append(mask_)
        mask_gray_batch.append(mask_gray_)
        pose_img_batch.append(pose_img_)

    pipe.to(device)
    pipe.unet_encoder.to(device)  
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                
                prompt = ["model is wearing " + garment_desc] * len(human_img_batch)
                negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(human_img_batch)
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = ["a photo of " + garment_desc] * len(human_img_batch)
                    
                    negative_prompt = ["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(human_img_batch)
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )
                    pose_img=[]
                    for pose in pose_img_batch:
                        pose_img.append(tensor_transform(pose).to(device, torch.float16))
                    pose_img=torch.stack(pose_img,0)
                    garm_tensor = tensor_transform(garm_img).unsqueeze(0).repeat(len(human_img_batch),1,1,1).to(device, torch.float16)
                    
                    
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device, torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                        cloth=garm_tensor.to(device, torch.float16),
                        mask_image=mask_batch,
                        image=human_img_batch,
                        height=global_height,
                        width=global_width,
                        ip_adapter_image=[garm_img.resize((global_width, global_height))]*len(human_img_batch),
                        guidance_scale=2.0,
                    )[0]

                    out_img_batch = []
                    for out_img in images:
                        out_img_batch.append(out_img)
                    return out_img_batch

def batch_preparation(is_checked, is_checked_crop, human_img_orig):
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((global_width, global_height))
    else:
        human_img = human_img_orig.resize((global_width, global_height))


    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        mask = mask.resize((global_width, global_height))
    else:
        mask = pil_to_binary_mask(human_img.resize((global_width, global_height)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    
    # if not os.path.exists('/root/autodl-tmp/new_IDM_VTON/ckpt/densepose/model_final_162be9.pkl'):
    load_file_from_url('https://hf-mirror.com/yisol/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl',os.path.join('./ckpt/densepose/'))
            

    args = apply_net.create_argument_parser().parse_args(('show', './densepose_rcnn_R_50_FPN_s1x.yaml',
                                                              './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((global_width, global_height))
    return human_img,mask,mask_gray,pose_img




garm_img_path = args.garm_img_path
data_path = args.data_path_imgs
    




import glob
files_list=sorted(glob.glob(f'{data_path}/*'))

files_list=files_list[::5]
random_files = random.sample(files_list, 4)

# Extract the numbers from the filenames
human_img_path = [int(f.split('_')[-1].split('.')[0]) for f in random_files]

# Write the numbers to a text file
with open('./selected_numbers.txt', 'w') as f:
    for number in human_img_path:
        f.write(f"{number}\n")

print("Selected file numbers written to selected_numbers.txt")
human_img_path=random_files

os.makedirs('./multi_first', exist_ok=True)
shutil.rmtree('./multi_first')
os.makedirs('./multi_first', exist_ok=True)



image_out = virtual_tryon(human_img_path, garm_img_path, garment_desc,denoise_steps=22,seed=0)


for idx, img in enumerate(image_out):
    img.save(f'./multi_first/image_out_{idx}.png')

transform = transforms.ToTensor()
tensor_images = [transform(img) for img in image_out]
torchvision.utils.save_image(tensor_images, f'./multi_first/grid.png')