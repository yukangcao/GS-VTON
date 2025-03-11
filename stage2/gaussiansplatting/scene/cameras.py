#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import torch
from torch import nn
import numpy as np
from gaussiansplatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal,getWorld2View2_tensor

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class Simple_Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, h, w,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", qvec=None
                 ):
        super(Simple_Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.qvec = qvec
        # print("Simple_Camera - FoVx :",FoVx)
        # print("Simple_Camera - FoVy :",FoVy)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = w
        self.image_height = h


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        
        self.sa=torch.tensor(getWorld2View2(R, T, trans, scale))

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def HW_scale(self, h, w):
        return Simple_Camera(self.colmap_id, self.R, self.T, self.FoVx, self.FoVy, h, w, self.image_name, self.uid ,qvec=self.qvec)


class C2W_Camera(nn.Module):
    def __init__(self, c2w, FoVx,FoVy, height, width,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", azimuth=None, elevation=None, dist=None,
                 ):
        super(C2W_Camera, self).__init__()
        # FoVx = focal2fov(fov2focal(FoVy, height), width)
        # FoVx = focal2fov(fov2focal(FoVy, width), height)

        # print("C2W_Camera - FoVx :",FoVx)
        # print("FoVy :",FoVy)

        w2c=c2w
        # w2c = torch.inverse(c2w)

        # # rectify...
        # w2c[1:3, :3] *= -1
        # w2c[:3, 3] *= -1

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height =height
        self.image_width = width

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.zfar = 100.0
        self.znear = 0.01

        # self.trans = trans.float()
        self.trans = trans
        self.scale = scale

        self.world_view_transform = w2c.transpose(0, 1).float().cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).float().cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).float()
        self.camera_center = self.world_view_transform.inverse()[3, :3].float()
        # print('self.camera_center',self.camera_center)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]



class Camera_HumanGaussian(nn.Module):
    def __init__(self, c2w, FoVy, height, width,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera_HumanGaussian, self).__init__()
        FoVx = focal2fov(fov2focal(FoVy, height), width)
        # FoVx = focal2fov(fov2focal(FoVy, width), height)

        w2c = torch.inverse(c2w)

        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height =height
        self.image_width = width

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans.float()
        self.scale = scale
        def getProjectionMatrix_2(znear, zfar, fovX, fovY):
            tanHalfFovY = math.tan((fovY / 2))
            tanHalfFovX = math.tan((fovX / 2))

            top = tanHalfFovY * znear
            bottom = -top
            right = tanHalfFovX * znear
            left = -right

            P = torch.zeros(4, 4)

            z_sign = 1.0

            P[0, 0] = 2.0 * znear / (right - left)
            P[1, 1] = 2.0 * znear / (top - bottom)
            P[0, 2] = (right + left) / (right - left)
            P[1, 2] = (top + bottom) / (top - bottom)
            P[3, 2] = z_sign
            P[2, 2] = z_sign * zfar / (zfar - znear)
            P[2, 3] = -(zfar * znear) / (zfar - znear)
            return P        

        # print('x1',w2c.shape)
        # print('x2',w2c.transpose(0, 1).shape)
        self.world_view_transform = w2c.transpose(0, 1).float().cuda()
        self.projection_matrix = getProjectionMatrix_2(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).float().cuda()
        # print('self.world_view_transform.unsqueeze(0)',self.world_view_transform.unsqueeze(0).shape)
        # print('x3',self.projection_matrix.unsqueeze(0).shape)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0).float()
        self.camera_center = self.world_view_transform.inverse()[3, :3].float()
        # print('self.camera_center',self.camera_center)