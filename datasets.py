### python lib
import os, sys, math, random, glob, cv2
import numpy as np

### torch lib
import torch
import torch.utils.data as data

### custom lib
import utils

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        
    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]


class MultiFramesDataset(data.Dataset):

    def __init__(self, opts, mode):
        super(MultiFramesDataset, self).__init__()

        self.opts = opts
        self.mode = mode
        self.ori_videos = []
        self.GT=[]
        self.MIDAS=[]
        self.num_frames = []
        if mode == 'train':
            root_path='data/train'
        else:
            root_path='data/test'

        self.videos=os.listdir(os.path.join(root_path,'MIDAS'))
        
        for video in self.videos:
            input_dir=os.path.join(root_path,'MIDAS',video)
            self.ori_videos.append(os.path.join(root_path,'video',video))
            if mode == 'train':
                self.GT.append(os.path.join(root_path,'GT',video))

            self.MIDAS.append(os.path.join(input_dir))
            frame_list = glob.glob(os.path.join(input_dir, '*.png'))

            if len(frame_list) == 0:
                raise Exception("No frames in %s" %input_dir)
                
            self.num_frames.append(len(frame_list))

        print("[%s] Total %d videos (%d frames)" %(self.__class__.__name__, len(self.MIDAS), sum(self.num_frames)))


    def __len__(self):
        return len(self.MIDAS)


    def __getitem__(self, index):

        ## random select starting frame index t between [0, N - #sample_frames]
        N = self.num_frames[index]
        T = random.randint(0, N - self.opts.sample_frames)

        ori_dir = self.ori_videos[index]
        input_dir = self.MIDAS[index]
        gt_dir=self.GT[index]
        ## sample from T to T + #sample_frames - 1
        frame_ori = []
        frame_gt=[]
        frame_d = []
        for t in range(T, T + self.opts.sample_frames):
            frame_d.append(utils.read_img(os.path.join(input_dir, f"{str(t)}-dpt_swin2_large_384.png"),grayscale=True))
        
        
        ## data augmentation
        if self.mode == 'train':
            for t in range(T, T + self.opts.sample_frames):
                frame_ori.append(utils.read_img(os.path.join(ori_dir, f"{str(t)}.png")))

            for t in range(T, T + self.opts.sample_frames):
                frame_gt.append(utils.read_img(os.path.join(gt_dir, f"{str(t)}.png"),grayscale=True))

            if self.opts.geometry_aug:
                ## random scale
                H_in = frame_d[0].shape[0]
                W_in = frame_d[0].shape[1]
                
                sc = np.random.uniform(self.opts.scale_min, self.opts.scale_max)
                H_out = int(math.floor(H_in * sc))
                W_out = int(math.floor(W_in * sc))

                ## scaled size should be greater than opts.crop_size
                if H_out < W_out:
                    if H_out < self.opts.crop_size:
                        H_out = self.opts.crop_size
                        W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
                else: ## W_out < H_out
                    if W_out < self.opts.crop_size:
                        W_out = self.opts.crop_size
                        H_out = int(math.floor(H_in * float(W_out) / float(W_in)))

                for t in range(self.opts.sample_frames):
                    frame_ori[t] = cv2.resize(frame_ori[t], (W_out, H_out))
                    frame_d[t] = np.expand_dims(cv2.resize(frame_d[t], (W_out, H_out)),2)
                    frame_gt[t] = np.expand_dims(cv2.resize(frame_gt[t], (W_out, H_out)),2)
                

            ## random crop
            cropper = RandomCrop(frame_d[0].shape[:2], (self.opts.crop_size, self.opts.crop_size))
            
            for t in range(self.opts.sample_frames):
                frame_ori[t] = cropper(frame_ori[t])
                frame_d[t] = cropper(frame_d[t])
                frame_gt[t]=cropper(frame_gt[t])
            

            if self.opts.geometry_aug:
                ### random rotate
                rotate = random.randint(0, 3)
                if rotate != 0:
                    for t in range(self.opts.sample_frames):
                        frame_ori[t] = np.rot90(frame_ori[t], rotate)
                        frame_d[t] = np.rot90(frame_d[t], rotate)
                        frame_gt[t] = np.rot90(frame_gt[t], rotate)
                    
                ## horizontal flip
                if np.random.random() >= 0.5:
                    for t in range(self.opts.sample_frames):
                        frame_ori[t] = cv2.flip(frame_ori[t], flipCode=0)
                        frame_d[t] = np.expand_dims(cv2.flip(frame_d[t], flipCode=0),2)
                        frame_gt[t] = np.expand_dims(cv2.flip(frame_gt[t], flipCode=0),2)
            if self.opts.order_aug:

                ## reverse temporal order
                if np.random.random() >= 0.5:
                    frame_ori.reverse()
                    frame_d.reverse()
                    frame_gt.reverse()
            #mirror repeate
            frame_d=frame_d[:-1]+frame_d[::-1] 
            frame_gt=frame_gt[:-1]+frame_gt[::-1]
            frame_ori=frame_ori[:-1]+frame_ori[::-1]
            ### convert (H, W, C) array to (C, H, W) tensor
            data = []
            for t in range(len(frame_d)):
                data.append(torch.from_numpy(frame_ori[t].transpose(2, 0, 1).astype(np.float32)).contiguous())
                data.append(torch.from_numpy(frame_d[t].transpose(2, 0, 1).astype(np.float32)).contiguous())
                data.append(torch.from_numpy(frame_gt[t].transpose(2, 0, 1).astype(np.float32)).contiguous())

        elif self.mode == "test":
            
            ## resize image to avoid size mismatch after downsampline and upsampling
            H_i = frame_d[0].shape[0]
            W_i = frame_d[0].shape[1]

            H_o = int(math.ceil(float(H_i) / self.opts.size_multiplier) * self.opts.size_multiplier)
            W_o = int(math.ceil(float(W_i) / self.opts.size_multiplier) * self.opts.size_multiplier)

            for t in range(self.opts.sample_frames):
                frame_d[t] = cv2.resize(frame_d[t], (W_o, H_o))
                frame_ori[t] = cv2.resize(frame_ori[t], (W_o, H_o))
                frame_gt[t] = cv2.resize(frame_gt[t], (W_o, H_o))
            #mirror
            frame_d=frame_d[:-1]+frame_d[::-1]
            frame_gt=frame_gt[:-1]+frame_gt[::-1]
            frame_ori=frame_ori[:-1]+frame_ori[::-1]
            data = []
            for t in range(len(frame_d)):
                data.append(torch.from_numpy(frame_d[t].transpose(2, 0, 1).astype(np.float32)).contiguous())
        
        else:
            raise Exception("Unknown mode (%s)" %self.mode)
 
        return data

