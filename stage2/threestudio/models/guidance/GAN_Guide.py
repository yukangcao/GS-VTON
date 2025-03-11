# from threestudio.models.guidance.generator_GAN import Discriminator
# from generator_GAN.stylegan2 import Discriminator
from generator_GAN.ganerf.generator.stylegan2 import Discriminator
# from threestudio.models.guidance.GAN_Guide import 
import torch
import numpy as np
import random
from torchvision.io import read_image
def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad
        
class GanerfModel:
    def __init__(self):
        self.discriminator = Discriminator(
                    block_kwargs={},
                    mapping_kwargs={},
                    epilogue_kwargs={"mbstd_group_size": 4},
                    channel_base=16384,
                    channel_max=512,
                    num_fp16_res=4,
                    conv_clamp=256,
                    img_channels=3,
                    c_dim=0,
                    img_resolution=64,
                )        
        
    
    def sample_crops_around_mask(image, mask, crop_size=(512, 512), num_crops=1):
        # Define transformations
        # transform = transforms_v2.Compose([
        #     transforms_v2.Resize((crop_size[1] * 2, crop_size[0] * 2)),  # Resize to include neighboring region
        #     transforms_v2.ToTensor()
        # ])

        # Read the image and the mask
        # image = Image.open(image_path)
        # mask = Image.open(mask_path)

        # Apply transformations
        # image = transform(image).unsqueeze(0)  # Add batch dimension
        # mask = transform(mask).unsqueeze(0)    # Add batch dimension
        # mask = transform(mask)    # Add batch dimension

        # Convert mask to binary
        mask = (mask > 0.5).float()

        # Get coordinates of masked region
        mask_indices = torch.nonzero(mask.squeeze(0)).numpy()
        print(mask_indices.shape)


        crop_samples = []
        for _ in range(num_crops):
            # Randomly select a point within the masked region
            idx = np.random.randint(0, mask_indices.shape[0])
            y, x = mask_indices[idx]
            print(y,x)

            # Ensure the sampled point is within image bounds
            x = max(min(x, image.size(3) - crop_size[0]), 0)
            y = max(min(y, image.size(2) - crop_size[1]), 0)

            # Extract the crop
            crop = image[:, :, y:y + crop_size[1], x:x + crop_size[0]]
            # crop_samples.append(crop)
            return crop

        return crop_samples
    
    def make_crop(self,images, resolution, times=1):
        # mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(1, times)
        mask, times = torch.ones_like(images[0:1, :, :]), times
        min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
        max_size = min(max_size, resolution - margin * 2)

        for _ in range(times):
            width = np.random.randint(int(min_size), int(max_size))
            height = np.random.randint(int(min_size), int(max_size))

            x_start = np.random.randint(int(margin), resolution - int(margin) - width + 1)
            y_start = np.random.randint(int(margin), resolution - int(margin) - height + 1)
            mask[:, y_start:y_start + height, x_start:x_start + width] = 0
            mask = 1 - mask
        

        # mask = 1 - mask if random.random() < 0.5 else mask
        
        return images[mask]
    
    def run_discriminator(self, x):
        bs, c, h, w = x.shape
        if self.training:
            input = x.contiguous()
        else:
            input = x
        # reshape
        d_img_resolution = self.discriminator.img_resolution
        if d_img_resolution < h:
            input = (
                input.unfold(2, d_img_resolution, d_img_resolution)
                .unfold(3, d_img_resolution, d_img_resolution)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(-1, c, d_img_resolution, d_img_resolution)
            )
        # call
        output = self.discriminator(input)
        # reshape
        if d_img_resolution < h:
            output = output.reshape(bs, 1, -1)
        return output
        
    def get_patches(self,real,fake):
        reals=self.sample_crops_around_mask(real)
        fakes=self.sample_crops_around_mask(fakes)
        return real,fakes
        
    
    def compute_simple_gradient_penalty(self, x):
        x.requires_grad_(True)
        pred_real = self.run_discriminator(x)
        gradients = torch.autograd.grad(outputs=[pred_real.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
        r1_penalty = gradients.square().sum([1, 2, 3]).mean()
        return r1_penalty / 2
        
    def compute_d_loss(self, real_patch, fake_patch):
        pred_real = self.run_discriminator(real_patch)
        pred_fake = self.run_discriminator(fake_patch.detach())
        loss_d_real = torch.nn.functional.softplus(-pred_real).mean()
        loss_d_fake = torch.nn.functional.softplus(pred_fake).mean()
        gradient_penalty = self.config.gp_loss_mult * self.compute_simple_gradient_penalty(real_patch)
        loss_d = loss_d_real + loss_d_fake
        return loss_d, gradient_penalty
    
        
    def get_discriminator_loss_dict(self,real_image,fake_image):
        loss_dict = {}
        if self.config.gp_loss_mult > 0:
            real_patches, fake_patches = self.get_patches(real_image,fake_image)
            # discriminator loss
            set_requires_grad(self.discriminator, True)
            loss_dict["d_loss"], loss_dict["gp"] = self.compute_d_loss(real_patches, fake_patches)
        return loss_dict
    
    
if __name__ == "__main__":
    a=GanerfModel()
    
    img=read_image('/root/autodl-tmp/Lora/realfill/data/cloth/ref/dsaas0.png')
    a.get_discriminator_loss_dict(img,img)