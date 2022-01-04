from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import os
import math
import cv2
from glob import glob

import easyocr
from task1.superglue.superpoint import SuperPoint
from task1.superglue.superglue import SuperGlue
from task1.superglue.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

class MatchImageSizeTo(object):
    def __init__(self, size=1080):
        self.size=size

    def __call__(self, img):
        H, W = img.shape

        if H>=W:
            W_size = int(W/H * self.size * (1920/1450))
            # W_size = int(W/H * self.size)
            img_new = cv2.resize(img, (W_size, self.size))
        else:
            H_size = int(H/W * self.size * (1450/1920))
            # H_size = int(H/W * self.size)
            img_new = cv2.resize(img, (self.size, H_size))
        
        return img_new

def ocr(frames, frame_idx_start, masked_frame_idx, texts, text_idx):
    reader = easyocr.Reader(['ko'], gpu=True)
    frame_num = len(frames)
    results = reader.readtext_batched(frames, 
                                    batch_size=frame_num, 
                                    output_format='dict', 
                                    blocklist='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZㆍ', 
                                    min_size = 5, 
                                    text_threshold=0.6)
    for i,result in enumerate(results):
        res = []
        for res_ in result:
            word = res_['text']
            if len(word) <= 2:
                continue
            if (word.endswith('실') or word.endswith('과')) and word[:3].isdigit():
                res.append({'text': word, 'confident':res_['confident']})
                continue
        if len(res) == 0:
            continue
        text = max(res, key=lambda x: x['confident'])['text']
        texts.append(text)
        text_idx.append(masked_frame_idx[frame_idx_start+i])

def match_pairs(vid_, imgs, vid_batch, device,
                match_num_rate_threshold=0.02,
                superglue='indoor', 
                max_keypoints = 1024, 
                keypoint_threshold = 0.0, 
                nms_radius = 4, 
                sinkhorn_iterations = 15, 
                match_threshold = 0.2):
    """ 
    Args: 
        vid_: list of numpy vid frames, range 0~255, shape H x W x 3 , BGR           
        imgs: list of numpy images, range 0~255, shape H x W , Grayscale
        vid_batch: batch size for video
    Return:
        result: list of tuples (frame idx, match_rate)
    """
    
    vid = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid_]

    torch.set_grad_enabled(False)

    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

    superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)
    superglue = SuperGlue(config.get('superglue', {})).eval().to(device)

    T = len(vid)
    N = len(imgs)
    
    imgs = [torch.from_numpy(imgs[i]/255.).float()[None,None] for i in range(N)]  # (1,1,H,W)
    imgs_kp = []
    match_num_threshold = []

    for img in imgs:
        img = img.to(device)
        kp = superpoint({'image': img})    # 'keypoints', 'scores', 'descriptors'
        kp = {**{k+'0': v for k, v in kp.items()}}
        for k in kp:
            if isinstance(kp[k], (list,tuple)):
                kp[k] = torch.stack(kp[k])    # (1,K,2), (1,K), (1,D,K)
        imgs_kp.append(kp)
        match_num_threshold.append(int(kp['keypoints0'].shape[1]*match_num_rate_threshold))

    result = [[-1,0] for i in range(N)]
    vid_size = vid[0].shape[-2:]
    Iters = math.ceil(T/vid_batch)
    start = 0

    from tqdm import tqdm
    with tqdm(total=Iters) as pbar:
        for i in range(Iters):
            start = i * vid_batch
            if i == Iters-1:
                end = T
            else:
                end = (i+1) * vid_batch
            frames = [torch.from_numpy(vid[i]/255.).float()[None] for i in range(start,end)] #(1,H,W)
            frames = torch.stack(frames).to(device)  # (B,1,H,W)
            vid_kp = superpoint({'image':frames})
            vid_kp = {**{k+'1': v for k, v in vid_kp.items()}}
            for k in vid_kp:
                if isinstance(vid_kp[k], (list,tuple)):
                    vid_kp[k] = torch.stack(vid_kp[k])    # (B,K,2), (B,K), (B,D,K)

            for n, img_kp_ in enumerate(imgs_kp):
                img_size = imgs[n].shape[-2:]
                img_kp = {}
                for k in img_kp_:
                    if len(img_kp_[k].shape)==2:
                        img_kp[k] = img_kp_[k].repeat((end-start),1) # (B,K,2), (B,K), (B,D,K)
                    else:
                        img_kp[k] = img_kp_[k].repeat((end-start),1,1) # (B,K,2), (B,K), (B,D,K)

                data = {**vid_kp, **img_kp, 'image0_shape': img_size, 'image1_shape': vid_size}
                pred = superglue(data)  # matches0, matches1, matching_scores0, matching_scores1
                pred = {k:v.cpu().numpy() for k,v in pred.items()} # all (B,~1024)
                match_num = np.sum(pred['matches0']>-1, axis=1) # (B,)
                max_idx = np.argmax(match_num)

                if match_num[max_idx] < match_num_threshold[n]:
                    continue
                elif match_num[max_idx] < result[n][1]:
                    continue
                else:
                    result[n][1] = match_num[max_idx]
                    result[n][0] = start + max_idx
            pbar.update(1)
    
    return result