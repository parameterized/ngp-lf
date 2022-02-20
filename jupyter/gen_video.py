
import os
import numpy as np
import quaternion
import torch
from torchvision import transforms
import moviepy.editor as mpy

from dataset import get_rays

def gen_batched_img(model, rays, batch_size):
    pred = torch.zeros_like(rays[:, :3])
    for i in range(int(np.ceil(rays.shape[0] / batch_size))):
        pred[i*batch_size : (i+1)*batch_size] = model(rays[i*batch_size : (i+1)*batch_size])
    return pred

def gen_video(dataset, model, path, duration=2, fps=12, cuda=True, batch_size=None):
    camxs = dataset.poses[:, 0, -1]
    camzs = dataset.poses[:, 2, -1]
    cam_left_id = np.argmin(camxs + camzs)
    cam_right_id = np.argmax(camxs - camzs)

    q_left = quaternion.from_rotation_matrix(dataset.poses[cam_left_id, :, :3])
    q_right = quaternion.from_rotation_matrix(dataset.poses[cam_right_id, :, :3])

    saw = lambda x: 1 - abs(x % 2 - 1)
    ease_quad = lambda x: 2 * x**2 if x < 0.5 else 1 - (-2 * x + 2)**2 / 2

    def make_frame(t):
        t = t * 2 / duration
        t = ease_quad(saw(t))
        
        a = np.array
        q_t = quaternion.quaternion_time_series.squad(a([q_left, q_right]), a([0, 1]), a([t]))[0]

        p_left = dataset.poses[cam_left_id, :, -1]
        p_right = dataset.poses[cam_right_id, :, -1]
        p_t = p_left * (1 - t) + p_right * t

        pose_t = np.concatenate([quaternion.as_rotation_matrix(q_t), p_t[:, None]], axis=-1).astype('float32')
        
        H, W, focal = dataset.H, dataset.W, dataset.focal
        with torch.no_grad():
            rays = get_rays(H, W, focal, pose_t)
            if cuda:
                rays = rays.cuda()
            if batch_size is None:
                pred = model(rays)
            else:
                pred = gen_batched_img(model, rays, batch_size)
            pred_r = transforms.Resize(256)(pred.clip(0, 1).reshape(H, W, 3).permute(2,0,1))
            img = pred_r.permute(1,2,0).multiply(255).int().cpu().numpy()
        return img

    os.makedirs(os.path.dirname(path), exist_ok=True)
    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_videofile(path, fps=fps)
