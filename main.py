from datetime import datetime
import gc
import subprocess
import time
import torch
import traceback
import glob
from pathlib import Path 
import os
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from diffusers.utils import load_image
from PIL import Image
os.environ['HF_HOME'] = '/mnt/sfs-common/ykcao/cache'

# download_dir = './downloaded_images'
# if not os.path.exists(download_dir):
#     os.makedirs(download_dir)

# Function to get the file extension from the URL
# def get_file_extension(url):
#     parsed_url = urlparse(url)
#     path = parsed_url.path
#     return unquote(path.split('.')[-1])

# Function to generate a unique filename based on the URL
# def get_filename_from_url(url):
#     hash_object = hashlib.md5(url.encode())
#     return f"{hash_object.hexdigest()}.{get_file_extension(url)}"

# Function to download images with caching
# def download_images(urls, save_dir, max_retries=2, timeout=2):
#     images = []
#     for url in urls:
#         try:
#             try:
#                 filename = get_filename_from_url(url)
#                 img_path = os.path.join(save_dir, filename)
#                 Image.open(url).save(img_path)       
#                 images.append(img_path)
#             except:
#                 filename = get_filename_from_url(url)
#                 img_path = os.path.join(save_dir, filename)
#                 load_image(url).save(img_path)       
#                 images.append(img_path)                    
#         except Exception as e:
        
#             print(f'Failed to download {url}: {e}')
        
#     return images

def process_images_and_generate_captions(cloth_path):
    

    # image_urls=[cloth_path]

    
    # downloaded_image_paths = download_images(image_urls, download_dir)
        # print('Found replica')

    
    # Initialize the model and processor
    use_cache=False
    output_file = "./image_descriptions.txt"
    if not use_cache:
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model_blip = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model_blip.to(device)
        

        prompt = "person is dressed in"
        
        # Generate captions for each image
        image = Image.open(cloth_path).convert('RGB')
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

        generated_ids = model_blip.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text


def process_images(image_path, data_path, height, weight, description,enable_idm_attn):
    """Process each image using new_IDM_VTON"""
    # Change to the IDM_VTON directory and run the Python script
    os.chdir('./stage1/')
    subprocess.run([
        'python3', './run_2d_tryon.py',
        '--garm_img_path', image_path,
        '--data_path_imgs', str(os.path.join(data_path, 'images')),
        f'--height={height}', f'--width={weight}',
        f'--garm_desc_given={description}'
    ])
    
    # Remove specific file
    os.remove("./multi_first/grid.png")

    
def run_realfill_pipeline(description):
    """Run the realfill training pipeline"""
    subprocess.run([
        'accelerate', 'launch', './train_lora.py',
        '--pretrained_model_name_or_path', 'stabilityai/stable-diffusion-2-inpainting',
        '--train_data_dir', './multi_first',
        '--output_dir', './models/pipeline_captureddata-model',
        '--resolution', '512', '--train_batch_size', '16', '--gradient_accumulation_steps', '1',
        '--unet_learning_rate', '2e-4', '--text_encoder_learning_rate', '4e-5',
        '--lr_scheduler', 'constant', '--lr_warmup_steps', '100', '--max_train_steps', '1000', '--validation_steps', '100',
        '--lora_rank', '8', '--lora_dropout', '0.1', '--lora_alpha', '16',
        '--enable_xformers_memory_efficient_attention', '--seed', '0',
        f'--garm_desc_given={description}'
    ])
    
    
def edit_gaussian_model(img_path, data_path, height, weight, gs_source, description,enable_ControlNet,enable_reInpaint,enable_attention,enable_limit,task_type):
    """Run GaussianEditor with the parsed data"""
        
    os.chdir('../stage2/')
    from ruamel.yaml import YAML

    # Define the YAML file path
    yaml_file = "./configs/idm_20.yaml"

    # Load the YAML file
    yaml = YAML()
    with open(yaml_file, 'r') as file:
        config = yaml.load(file)

    # Modify the width and height in the YAML content
    weight = int(weight)  # Replace with your weight value
    height = int(height)  # Replace with your height value

    config['data']['width'] = weight
    config['data']['height'] = height

    with open(yaml_file, 'w') as file:
        yaml.dump(config, file)        

    # Create timestamp
    base_name = os.path.basename(img_path)
    model = data_path.split('/')[-1]
    timestamp = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model}_{os.path.splitext(base_name)[0]}"
    
    # Run GaussianEditor
    subprocess.run([
        'python3', 'launch.py',
        f'timestamp={timestamp}',
        '--config', './configs/idm_20.yaml',
        '--train',
        '--gpu', '0',
        'trainer.max_steps=4000',
        'system.prompt_processor.prompt="x"',
        'system.max_densify_percent=0.03',
        'system.anchor_weight_init_g0=0.0',
        'system.anchor_weight_init=0.02',
        'system.anchor_weight_multiplier=1.3',
        'system.seg_prompt="man"',
        'system.loss.lambda_anchor_color=5',
        'system.loss.lambda_anchor_geo=50',
        'system.loss.lambda_anchor_scale=50',
        'system.loss.lambda_anchor_opacity=50',
        'system.densify_from_iter=100',
        'system.densify_until_iter=5000',
        'system.densification_interval=300',
        f'data.source={data_path}',
        f'system.gs_source={gs_source}',
        'system.loggers.wandb.enable=false',
        'system.loggers.wandb.name="edit_n2n_face_Ein"',
        f'system.enable_attention={enable_attention}',
        f'system.enable_ControlNet={enable_ControlNet}',
        'system.second_phase=false',
        f'system.enable_reInpaint={enable_reInpaint}',
        f'system.inpaint_prompt={description}',
        f'system.control_image_path={img_path}',
        'system.inpaint_model_path="../stage1/models/pipeline_captureddata-model"',
        'system.renew_render_interval=1000',
        f'system.global_height={int(height)}',
        f'system.global_width={int(weight)}',
        'system.which_step=0',
        f'system.enable_limit={enable_limit}',
    ], check=True)



        
Tasks={
    'gsvton':{'idm_attn':'true','inpaint_attn':'true','controlnet':'true','iterative':'False'},
}


previous_idm_attn = {'prev': None }

def Run(args):        
        
        
        generated_text = process_images_and_generate_captions(args.cloth_path)
        weight, height=Image.open(glob.glob(os.path.join(args.data_path,'images/*.png'))[0]).size

        image_paths = args.cloth_path
        data_path = args.data_path
        gs_source = args.gs_source 
        

        
        time.sleep(10)
        torch.cuda.empty_cache()
        gc.collect()
        
        for key,value in Tasks.items():
            
            
            desc_IDM=generated_text

            if previous_idm_attn['prev']!=value['idm_attn']:
                process_images(image_paths, data_path, height, weight, description=desc_IDM,enable_idm_attn=value['idm_attn'])
                time.sleep(10)
                torch.cuda.empty_cache()
                run_realfill_pipeline(description=generated_text)
                torch.cuda.empty_cache()
                time.sleep(10)
            
            previous_idm_attn['prev']=value['idm_attn']
            torch.cuda.empty_cache()
            edit_gaussian_model(image_paths, data_path, height, weight, gs_source,generated_text,enable_ControlNet=value['controlnet'],enable_reInpaint=value['iterative'],enable_attention=value['inpaint_attn'],enable_limit=True,task_type=key)        
            

if __name__=="__main__":
    
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--gs_source',type=str)
    parser.add_argument('--cloth_path',type=str)
    parser.add_argument('--data_path',type=str)
    args=parser.parse_args()
    
    Run(args)