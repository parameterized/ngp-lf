
import os
import json
import numpy as np
import torch
from torch.nn.functional import grid_sample
from torchvision import transforms
from PIL import Image

BICUBIC = transforms.functional.InterpolationMode.BICUBIC


# https://github.com/Fyusion/LLFF

def load_poses_bounds_llff(scenedir):
    poses_arr = np.load(os.path.join(scenedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    
    return poses, bds, hwf

def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1]
    return rays_o, rays_d


def get_rays(H, W, focal, c2w):
    ro, rd = get_rays_np(H, W, focal, c2w)
    rd = torch.tensor(rd).reshape(-1, 3)
    ro = torch.tensor(ro).reshape(-1, 3).broadcast_to(rd.shape)
    rd = rd / rd.norm(dim=-1, keepdim=True)
    return torch.cat([ro, rd], axis=-1)


class Dataset:
    """Load scene data
    
    Args:
        scene_path: path to scene
        scene_type: 'llff' or 'blender'
        resize: None or length for smaller edge (default is 256)
        parition: 'train', 'val', or 'test' (for blender scenes, default is 'train')
    """
    def __init__(self, scene_path, scene_type, resize=256, partition='train'):
        if resize is None:
            transform = transforms.ToTensor()
        else:
            transform = transforms.Compose([
                transforms.Resize(resize, interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.clip(0, 1))
            ])

        if scene_type == 'llff':
            img_folder = os.path.join(scene_path, 'images')
            self.images = [transform(Image.open(os.path.join(img_folder, v)))
                for v in sorted(os.listdir(img_folder))]
            self.images = torch.stack(self.images, axis=0)
            self.poses, self.bds, hwf = load_poses_bounds_llff(scene_path)
            self.H, self.W = self.images[0].shape[1:]
            self.focal = self.H / hwf[0] * hwf[2]
        elif scene_type == 'blender':
            with open(os.path.join(scene_path, f'transforms_{partition}.json'), 'r') as f:
                transforms_json = json.load(f)
            self.images = []
            self.poses = []
            self.bds = []
            for frame in transforms_json['frames']:
                img = Image.open(os.path.join(scene_path, frame['file_path'][2:] + '.png'))
                self.images.append(transform(img))
                self.poses.append(np.array(frame['transform_matrix'], dtype='float32')[:3, :])
                self.bds.append(np.array([2, 6], dtype='float32'))
            self.images = torch.stack(self.images, axis=0)
            if self.images.shape[1] == 4:
                self.images = self.images[:, :3, ...] * self.images[:, 3:4, ...]
            self.poses = np.stack(self.poses, axis=0)
            self.bds = np.stack(self.bds, axis=0)
            self.H, self.W = self.images[0].shape[1:]
            self.focal = .5 * self.W / np.tan(.5 * transforms_json['camera_angle_x'])

            # fix axes and rescale
            _poses = self.poses.copy()
            self.poses[:, 1, :] = -_poses[:, 2, :]
            self.poses[:, 2, :] = _poses[:, 1, :]
            self.poses[:, :3, -1] *= 5
            self.bds *= 5
        else:
            raise ValueError(f"Type {scene_type} not one of 'llff' or 'blender'")

        print('Camera position mean & std:')
        print(self.poses[:,:,-1].mean(axis=0), self.poses[:,:,-1].std(axis=0))

        rays = []
        colors = []

        for i in range(len(self.poses)):
            rays.append(get_rays(self.H, self.W, self.focal, self.poses[i]))
            colors.append(self.images[i].permute(1,2,0).reshape(-1, 3))

        self.rays = torch.cat(rays, axis=0).cuda()
        self.colors = torch.cat(colors, axis=0).cuda()
    
    def idxy_from_ray_ids(self, ray_ids):
        """Get a tensor of [image id, x, y] from a tensor of ray indices.
        x and y are in the range [0, 1]
        """
        idxy = torch.zeros(ray_ids.shape[0], 3).cuda()
        idxy[:, 0] = ray_ids // (self.H * self.W)
        idxy[:, 1] = ray_ids / self.W % 1
        idxy[:, 2] = ray_ids // self.W / self.H - idxy[:, 0]
        return idxy


class MultiscaleDataset(Dataset):
    """Load scene data
    
    Args:
        scene_path: path to scene
        scene_type: 'llff' or 'blender'
        resize: None or length for smaller edge (default is 256)
        parition: 'train', 'val', or 'test' (for blender scenes, default is 'train')
    """
    def __init__(self, scene_path, scene_type, resize=256, partition='train'):
        super().__init__(scene_path, scene_type, resize, partition)
        self.scaled_images = [self.images]
        size = min(self.H, self.W)
        while size > 1:
            size //= 2
            rs_transform = transforms.Resize(size, interpolation=BICUBIC)
            rs_imgs = rs_transform(self.scaled_images[-1]).clip(0, 1)
            self.scaled_images.append(rs_imgs)
        
        self.images_full = torch.zeros(
            3, self.H * self.images.shape[0], int(np.ceil(self.W * 3 / 2))).cuda()
        self.images_full[:, :, :self.W] = torch.cat(
            [*self.scaled_images[0]], dim=1)
        
        x = self.W
        y_offset = 0
        for i, imgs in enumerate(self.scaled_images[1:]):
            h, w = imgs.shape[-2:]
            for j, img in enumerate(imgs):
                y = j * self.H + y_offset
                self.images_full[:, y:y+h, x:x+w] = img
            y_offset += h
    
    def sample(self, x):
        """Sample images with a tensor containing
        [image id, pixel size, x, y] on its last dimension

        pixel size, x, and y are all in the range [0, 1]
        """
        c1 = x[..., 2:].clone()
        c2 = c1.clone()
        
        z0 = torch.zeros(1).cuda()
        base_size = torch.tensor([min(self.H, self.W)]).cuda()
        imax = torch.log2(base_size).floor()
        i = torch.log2(x[..., 1] * base_size)
        i1 = i.floor().clip(z0, imax)
        i2 = i.ceil().clip(z0, imax)
        s1 = torch.pow(0.5, i1)
        s2 = torch.pow(0.5, i2)
        
        x_off_1 = (i1 > 0.5).float()
        x_off_2 = (i2 > 0.5).float()
        y_off_1 = torch.maximum(1 - 2 * s1, z0) + x[..., 0]
        y_off_2 = torch.maximum(1 - 2 * s2, z0) + x[..., 0]
        
        c1[..., -2] = (c1[..., -2] * s1 + x_off_1) * (4 / 3) - 1
        c2[..., -2] = (c2[..., -2] * s2 + x_off_2) * (4 / 3) - 1
        c1[..., -1] = (c1[..., -1] * s1 + y_off_1) * (2 / len(self.images)) - 1
        c2[..., -1] = (c2[..., -1] * s2 + y_off_2) * (2 / len(self.images)) - 1
        
        rgb1 = grid_sample(self.images_full.unsqueeze(0), c1,
            mode='bilinear', padding_mode='border')
        rgb2 = grid_sample(self.images_full.unsqueeze(0), c2,
            mode='bilinear', padding_mode='border')
        
        t1 = torch.pow(2, i1) / base_size
        t2 = torch.pow(2, i2) / base_size
        td = t2 - t1
        td[td == 0] = 1
        t = torch.pow(2, i) / base_size
        t = (t - t1) / td
        return torch.lerp(rgb1, rgb2, t)