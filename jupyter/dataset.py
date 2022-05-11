
import os
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


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
                transforms.Resize(resize),
                transforms.ToTensor()
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
