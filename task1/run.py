import os
from threading import Thread
import sys
import time
import cv2
import numpy
import argparse
import math
from glob import glob
import numpy as np
import json
import torch

from task1.task1_utils import match_pairs, MatchImageSizeTo,ocr

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="0"



def task1(frames, imgs, search_radius=30, ocr_batch_size=10, match_batch_size=8):
    frame_total = len(frames)

    ######################### Frame-Image Matching ###########################
    print('start image matching')
    match_results = match_pairs(frames, imgs, match_batch_size, 'cuda')
    torch.cuda.empty_cache()

    ######################### mask frames ###########################
    print('masking')
    vid_mask = np.zeros(frame_total).astype(np.int)
    img_idx = []
    for match_res in match_results:
        img_idx.append(match_res[0])
        if match_res[0] == -1:
            continue
        idx = np.arange(search_radius*2+1) - search_radius + match_res[0]
        idx = np.clip(idx,0,frame_total-1).astype(np.int)
        vid_mask[idx] = 1
    masked_frame_idx = np.where(vid_mask==1)[0]
    frames = np.stack(frames, axis=0)[vid_mask==1]

    ######################### OCR ###########################
    print('start ocr')
    texts = []
    text_idx = []
    Iters = math.ceil(masked_frame_idx.shape[0]/ocr_batch_size)
    from tqdm import tqdm
    with tqdm(total=Iters) as pbar:
        for i in range(Iters):
            start =  i*ocr_batch_size
            if i == Iters-1:
                end = masked_frame_idx.shape[0]
            else:
                end = (i+1)*ocr_batch_size
            ocr(frames[start:end], start, masked_frame_idx, texts, text_idx)
            pbar.update(1)
    torch.cuda.empty_cache()

    answer = []
    for i, i_idx in enumerate(img_idx):
        ans = 'NONE'
        if i_idx == -1:
            answer.append(ans)
            continue
        min_dist = search_radius+1
        for j,t_idx in enumerate(text_idx):
            if abs(t_idx-i_idx) > search_radius:
                continue
            if abs(t_idx-i_idx) < min_dist:
                min_dist = abs(t_idx-i_idx)
                ans = texts[j]
        answer.append(ans)

    print(answer)

    return answer

def task1_main(video_path, img_path, frame_skip):
    #print(video_path)
    start = time.time()
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--video_path', default='/home/jaewon/drone/samples', help='video path')
    #parser.add_argument('--img_path', default='/home/jaewon/drone/samples', help='image path')
    #parser.add_argument('--output_path', default='output.json', help='output path')
    #parser.add_argument('--frame_skip', type=int, default=30, help='output path')
    #args = parser.parse_args()

    # f = open(args.output_path,'w')
    final_result = {
                        "task1_answer":[{
                            "set_1": [],
                            "set_2": [],
                            "set_3": [],
                            "set_4": [],
                            "set_5": []
                        }]
                    }

    imgs=[]

    img_list = glob(os.path.join(img_path, "*.jpg"))
    #print(img_list)
    img_list.sort()
    for img_ in img_list:
        if "rescue" in img_ :
            img = cv2.imread(img_, cv2.IMREAD_GRAYSCALE)
            img = MatchImageSizeTo()(img)
            imgs.append(img)
    
    vid_list = glob(os.path.join(video_path, "*.mp4"))
    vid_list.sort()
    #print("--------------------")
    #print(vid_list, img_list)
    for vid_path in vid_list:
        vid_name = vid_path.split('/')[-1].split('.')[0].split('_')
        set_num = "set_{}".format(vid_name[0][-1])
        drone_num = "drone_{}".format(vid_name[1][-1])

        frames = []
        cap = cv2.VideoCapture(vid_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if(type(frame) == type(None)):
                break
            if frame_pos % frame_skip != 0 :
                continue
            frames.append(frame)
        cap.release()
        result = task1(frames, imgs)
        final_result["task1_answer"][0][set_num].append({drone_num:result})

    print(final_result)
    #with open(args.output_path, 'w') as f:
    #    json.dump(final_result, f)

    print("TASK1 TIME :", time.time()-start)

    return final_result

