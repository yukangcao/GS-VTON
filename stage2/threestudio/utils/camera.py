import torch

from gaussiansplatting.utils.graphics_utils import fov2focal
from gaussiansplatting.scene.cameras import Simple_Camera
import torch.nn.functional as F
from threestudio.utils.typing import *
import numpy as np

def camera_ray_sample_points(
    camera,
    scene_radius: float,
    n_points: int = 256,
    mask: Bool[Tensor, "H W"] = None,
    sampling_method: str = "inbound",
) -> Float[Tensor, "N n_points 3"]:
    fx = fov2focal(camera.FoVx, camera.image_width)
    fy = fov2focal(camera.FoVy, camera.image_height)

    # Fuck this shit transpose
    c2w = torch.inverse(camera.world_view_transform.T)
    # c2w = camera.world_view_transform
    R = c2w[:3, :3]
    T = c2w[:3, 3]

    camera_space_ij = torch.meshgrid(
        torch.arange(camera.image_width, dtype=torch.float32),
        torch.arange(camera.image_height, dtype=torch.float32),
        indexing="xy",
    )
    camera_space_ij = torch.stack(camera_space_ij, dim=-1)

    if mask is None:
        camera_space_ij = camera_space_ij.reshape(-1, 2)
    else:
        camera_space_ij = camera_space_ij[mask]

    assert camera_space_ij.ndim == 2

    camera_space_ij = (
        camera_space_ij
        - torch.tensor([[camera.image_width, camera.image_height]], dtype=torch.float32)
        / 2
    )

    view_space_xy = camera_space_ij * torch.tensor([[1 / fx, 1 / fy]])
    view_space_xyz = torch.cat(
        [view_space_xy, torch.ones_like(view_space_xy[..., 0:1])], dim=-1
    )

    view_space_directions = torch.bmm(
        R[None, ...].repeat(view_space_xyz.shape[0], 1, 1), view_space_xyz[..., None]
    )[..., 0]
    view_space_xyz = view_space_directions + T[None, ...]

    distances = None
    if sampling_method == "inbound":
        distances = torch.linspace(0, scene_radius * 2, n_points)
    elif sampling_method == "segmented":
        # linear inside scene radius, linear disparity outside, I forget the exact name of this sampling strategy
        distances_inside = torch.linspace(0, scene_radius, n_points // 2)
        distances_outside = torch.linspace(0, 1, n_points // 2)
    else:
        raise ValueError(f"Unknown sampling method {sampling_method}")

    return (
        view_space_directions[..., None, :] * distances[None, ..., None]
        + view_space_xyz[..., None, :]
    )


def project(camera: Simple_Camera, points3d):
    # TODO: should be equivalent to full_proj_transform.T
    if isinstance(points3d, list):
        points3d = torch.stack(points3d, dim=0)
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
    xy = points3d_camera[..., :2] / points3d_camera[..., 2:]
    ij = (
        xy
        * torch.tensor(
            [
                fov2focal(camera.FoVx, camera.image_width),
                fov2focal(camera.FoVy, camera.image_height),
            ],
            dtype=torch.float32,
            device=xy.device,
        )
        + torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=xy.device,
        )
        / 2
    ).to(torch.long)

    return ij




def unproject_ax(camera: Simple_Camera, points2d, depth):
    origin = camera.camera_center.cpu()
    w2c = camera.world_view_transform.T.cpu()
    R = w2c[:3, :3].T

    if isinstance(points2d, (list, tuple)):
        points2d = torch.stack(points2d, dim=0)

    print('x1',points2d)
    # points2d[0] *= camera.image_width
    # points2d[1] *= camera.image_height
    print('x2',points2d)
    points2d = points2d.to(w2c.device)
    points2d = points2d.to(torch.long)

    directions = (
        points2d
        - torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=w2c.device,
        )
        / 2
    ) / torch.tensor(
        [
            fov2focal(camera.FoVx, camera.image_width),
            fov2focal(camera.FoVy, camera.image_height),
        ],
        dtype=torch.float32,
        device=w2c.device,
    )
    padding = torch.ones_like(directions[..., :1])
    directions = torch.cat([directions, padding], dim=-1)
    if directions.ndim == 1:
        directions = directions[None, ...]
    directions = torch.einsum("ij,bj->bi", R, directions)
    directions = F.normalize(directions, dim=-1)
    print(  'x',  depth[points2d[..., 1], points2d[..., 0]].unsqueeze(0).shape )
    print(   'y',directions.shape )
    points3d = (
        directions.T @ depth[points2d[..., 1], points2d[..., 0]].unsqueeze(0).T + origin[None, ...]
    )

    return points3d

def rgb_depth_to_point_cloud(camera, rgb, depth):
    h, w = depth.shape
    x = torch.arange(0, w)
    y = torch.arange(0, h)
    xv, yv = torch.meshgrid(x, y)
    points2d = torch.stack((xv.flatten(), yv.flatten()), dim=1).reshape(-1,2)
    # points2d=torch.tensor([[5,5],[1,1],[300,1]]).reshape(-1,2)
    # points2d = torch.stack((xv.flatten(), yv.flatten()), dim=1).reshape(-1,2)
    # points2d=torch.tensor([5,5]).reshape(2)
    points3d = unproject_ax(camera, points2d, depth)
    # rgb_flat = rgb.view(-1, 3)
    # point_cloud = torch.cat([points3d.view(-1, 3), rgb_flat], dim=1)
    print('points3d:',points3d.shape)
    return points3d




def unproject_Mine(camera: Simple_Camera, depth_map):
    
    depth=depth_map
    origin = camera.camera_center
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3].T

    height, width = depth_map.shape
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)
    
    
    points2d = torch.from_numpy(np.stack((x, y), axis=-1)).to(w2c.device).long()
    

    directions = (points2d - torch.tensor([camera.image_width, camera.image_height], dtype=torch.float32, device=w2c.device) / 2) / \
                 torch.tensor([fov2focal(camera.FoVx, camera.image_width), fov2focal(camera.FoVy, camera.image_height)], dtype=torch.float32, device=w2c.device)

    padding = torch.ones_like(directions[..., :1])
    directions = torch.cat([directions, padding], dim=-1)

    
    directions = torch.einsum("ij,abj->abi", R, directions)
    directions = F.normalize(directions, dim=-1)
    

    
    # points3d = directions * depth[points2d[..., 1], points2d[..., 0]].unsqueeze(2)+ origin[None, ...]
    points3d = directions * depth[points2d[..., 1], points2d[..., 0]].unsqueeze(2)+ origin

    return points3d.reshape(-1,3)


def unproject_sa(camera: Simple_Camera, points2d, depth):
    origin = camera.camera_center
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3].T

    if isinstance(points2d, (list, tuple)):
        points2d = torch.stack(points2d, dim=0)

    # points2d[0] *= camera.image_width
    # points2d[1] *= camera.image_height
    points2d = points2d.to(w2c.device)
    points2d = points2d.to(torch.long)

    directions = (
        points2d
        - torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=w2c.device,
        )
        / 2
    ) / torch.tensor(
        [
            fov2focal(camera.FoVx, camera.image_width),
            fov2focal(camera.FoVy, camera.image_height),
        ],
        dtype=torch.float32,
        device=w2c.device,
    )
    padding = torch.ones_like(directions[..., :1])
    directions = torch.cat([directions, padding], dim=-1)
    if directions.ndim == 1:
        directions = directions[None, ...]
    directions = torch.einsum("ij,bj->bi", R, directions)
    directions = F.normalize(directions, dim=-1)

    points3d = (
        directions * depth[points2d[..., 1], points2d[..., 0]] + origin[None, ...]
    )

    return points3d

def rgb_depth_to_point_cloud(camera, rgb, depth):
    out=[]
    for i in range(512):
        for j in range(512):
            out_temp= unproject_sa(camera,torch.tensor([i,j]).reshape(2),torch.rand((512, 512)))
            out.append(out_temp)
    
    # a=torch.tensor([2,5]).reshape(2)
    # x = np.linspace(0, width - 1, width)
    # y = np.linspace(0, height - 1, height)
    # x, y = np.meshgrid(x, y)
        
    # out= unproject_sa(camera,a,torch.rand((512, 512))  )
    print("Out:",torch.stack(out).shape)
    return out


def unproject(camera: Simple_Camera, points2d, depth):
    origin = camera.camera_center
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3].T

    if isinstance(points2d, (list, tuple)):
        points2d = torch.stack(points2d, dim=0)

    points2d[0] *= camera.image_width
    points2d[1] *= camera.image_height
    points2d = points2d.to(w2c.device)
    points2d = points2d.to(torch.long)

    directions = (
        points2d
        - torch.tensor(
            [camera.image_width, camera.image_height],
            dtype=torch.float32,
            device=w2c.device,
        )
        / 2
    ) / torch.tensor(
        [
            fov2focal(camera.FoVx, camera.image_width),
            fov2focal(camera.FoVy, camera.image_height),
        ],
        dtype=torch.float32,
        device=w2c.device,
    )
    padding = torch.ones_like(directions[..., :1])
    directions = torch.cat([directions, padding], dim=-1)
    if directions.ndim == 1:
        directions = directions[None, ...]
    directions = torch.einsum("ij,bj->bi", R, directions)
    directions = F.normalize(directions, dim=-1)

    points3d = (
        directions * depth[0][points2d[..., 1], points2d[..., 0]] + origin[None, ...]
    )

    return points3d


def get_point_depth(points3d, camera: Simple_Camera):
    w2c = camera.world_view_transform.T
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
    depth = points3d_camera[..., 2:]
    return depth
