# -*- coding: utf-8 -*-
import argparse
import cv2
import importlib
import numpy as np
import os
import torch

from core.utils import Stack, ToTorchFormatTensor
from PIL import Image
from torchvision import transforms


parser = argparse.ArgumentParser(description="STTN")
parser.add_argument("-v", "--video", type=str, required=True)
parser.add_argument("-m", "--mask",   type=str, required=True)
parser.add_argument("-c", "--ckpt",   type=str, required=True)
parser.add_argument("--model",   type=str, default='sttn')
parser.add_argument("-s", "--img_sequence", action='store_true')
args = parser.parse_args()


w, h = 432, 240
ref_length = 10
neighbor_stride = 5
default_fps = 24

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


# read frame-wise masks 
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        m = m.resize((w, h), Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4) # from original sttn
        masks.append(Image.fromarray(m*255))
    return masks

def read_images(impath):    
    images = []
    imnames = os.listdir(impath)
    imnames.sort()
    for im in imnames: 
        im = Image.open(os.path.join(impath, im))
        im = im.resize((w,h))
        images.append(im)
    return images

#  read frames from video 
def read_frame_from_videos(vname):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((w,h)))
        success, image = vidcap.read()

    return frames       


def main_worker(args_video, video_name, args_mask, mask_name):

    if (args_video == video_name):
        video_name = ""

    # if reading from a folder with images (extracted from a video):
    if (args.img_sequence):
        print("Reading from a sequence of images...")
        frames = read_images(args_video)

    # if reading from a video file:
    if (not args.img_sequence):
        print("Reading from a video file...")
        frames = read_frame_from_videos(args_video)

    # set up models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(model_path))
    model.eval()
    
    video_length = len(frames)
    feats = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    masks = read_mask(args_mask)
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    feats, masks = feats.to(device), masks.to(device)
    comp_frames = [None]*video_length

    with torch.no_grad():
        # (1-masks)   -> what is 0 in the mask becomes 1. what is 255 in the mask becomes 0
        # feats*(...) -> makes the object highlighted in the video becomes all black in feats
        feats = model.encoder((feats*(1-masks).float()).view(video_length, 3, h, w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)
    print('loading videos and masks from: {}'.format(args_video))

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(neighbor_ids, video_length)
        with torch.no_grad():
            pred_feat = model.infer(feats[0, neighbor_ids+ref_ids, :, :, :], masks[0, neighbor_ids+ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)*binary_masks[idx] + frames[idx] * (1-binary_masks[idx])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5
    writer = cv2.VideoWriter(f"{video_name.split('.')[0].split('/')[-1]}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
    #writer = cv2.VideoWriter(f"{args_video[:-len(video_name)-3]+video_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
    for f in range(video_length):
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*binary_masks[f] + frames[f] * (1-binary_masks[f])
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(f"{video_name.split('.')[0].split('/')[-1]}.mp4"))



if __name__ == '__main__':
    
    if (args.img_sequence):
        vidList = os.listdir(args.video)
        maskList = os.listdir(args.mask)
        vidList.sort()
        maskList.sort()        

        for video, mask in zip(vidList, maskList):
            if (video == mask):
                main_worker(args.video+video, video, args.mask+mask, mask)     
    
    main_worker(args.video, args.video, args.mask, args.mask)
