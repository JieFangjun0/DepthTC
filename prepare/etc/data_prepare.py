import torch, torchvision #该数据生成方法采用了插值的方法
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os
from bitarray import bitarray
import argparse
from glob import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


#cmp lib
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import yaml
import os
import cv2
import random

from glob import glob

from CMP.utils import flowlib
from CMP import models 
from CMP import utils
import importlib
importlib.reload(utils)

#flow warp
import sys
sys.path.append('/home/sam/DepthTC')
sys.path.append('/home/sam/DepthTC/pwc')
print(sys.path)
from pwc import Resample2d
flow_warping = Resample2d().to('cuda')
# load the parameters of model
class ArgObj(object):
    def __init__(self):
        pass

exp = '/home/sam/DepthTC/prepare/etc/CMP/experiments/semiauto_annot/resnet50_vip+mpii_liteflow'
load_iter = 42000
configfn = "{}/config.yaml".format(exp)

args = ArgObj()
with open(configfn) as f:
    config = yaml.safe_load(f)

for k, v in config.items():
    setattr(args, k, v)

setattr(args, 'load_iter', load_iter)
setattr(args, 'exp_path', os.path.dirname(configfn))

model = models.__dict__[args.model['arch']](args.model, dist_model=False)
model.load_state("{}/checkpoints".format(args.exp_path), args.load_iter, False)
model.switch_to('eval')

data_mean = args.data['data_mean']
data_div = args.data['data_div']

img_transform = transforms.Compose([transforms.Normalize(data_mean, data_div)])
depth_mean=[97.75]
depth_div=[53.79]
depth_transform=transforms.Compose([transforms.Normalize(depth_mean, depth_div)])
fuser = utils.Fuser(args.model['module']['nbins'], args.model['module']['fmax'])
torch.cuda.synchronize()






# Select a model and its config file from the model zoo
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("Cityscapes/mask_rcnn_R_50_FPN.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Cityscapes/mask_rcnn_R_50_FPN.yaml")

print('....model loading....')
predictor = DefaultPredictor(cfg)
print('model loaded')

# Put clean images you want to predict masks for here
clean_images_dir="/home/sam/Pictures/HR-WSI/train/imgs/"
# clean_images_dir="/home/sam/DepthTC/prepare/etc/examples/gt/breakdance/"
clean_depth_dir="/home/sam/Pictures/HR-WSI/train/gts/"
rootdir = '/home/sam/DepthTC/data/train'

video_length=10

clean_images = os.listdir(clean_images_dir)
clean_images.sort()
clean_images=clean_images[12238:][4443:]
clean_depths = os.listdir(clean_depth_dir)
clean_depths.sort()
clean_depths=clean_depths[12238:][4443:]
videopaths = []
depthpaths = []
for f in clean_images:
    videopaths.append(os.path.join(rootdir,'video',f.split('.')[0]))
    depthpaths.append(os.path.join(rootdir,'depth',f.split('.')[0]))
saved_num=0
for i in range(len(clean_images)):
    videopath = videopaths[i]
    depthpath = depthpaths[i]
    image = cv2.imread(os.path.join(clean_images_dir,clean_images[i]))
    depth=cv2.imread(os.path.join(clean_depth_dir,clean_depths[i]),cv2.IMREAD_GRAYSCALE)

    outputs = predictor(image)
    # Obtain Class_ids, Masks and Confidence from Outputs
    r = outputs['instances'].to('cpu')
    ids = r.pred_classes
    masks = r.pred_masks
    scores = r.scores
    # Filter out instances with a confidence lower than your threshold, e.g., 85%
    masks = masks[scores>0.90].numpy().transpose([1,2,0])
    ids = ids[scores>0.90]
    scores = scores[scores>0.90]
    
    if ids.shape[0]==0:
        continue
    mask_size=[masks[:,:,i].sum() for i in range(masks.shape[2])]
    flag=False
    mask_index=[]
    for s in range(len(mask_size)):
        if mask_size[s]>10000:
            flag=True
            mask_index.append(s)

    if not flag: #没有足够大的物体
        continue

    first_flag=False
    if not os.path.exists(videopath):
        os.makedirs(videopath)
    if not os.path.exists(depthpath):
        os.makedirs(depthpath)
    cv2.imwrite(os.path.join(videopath,'0.png'),image)
    cv2.imwrite(os.path.join(depthpath,'0.png'),depth)
    repeat = 1
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    tensor = img_transform(torch.from_numpy(np.array(image).astype(np.float32).transpose((2,0,1))))
    image = tensor.unsqueeze(0).repeat(repeat,1,1,1).cuda()

    depth=np.expand_dims(depth, axis=0)
    tensor =depth_transform(torch.from_numpy(depth.astype(np.float32)))
    depth = tensor.unsqueeze(0).cuda()
    

    seg=masks
    size = image.size

    # Randomly sample vectors
    coords = [] #[instance[sample[video_length[...]...]...]...]

    if seg.shape[0] != 0:
        for j in mask_index:
            m = seg[:,:,j]
            (x, y) = np.where(m==1)
            xy = np.concatenate([np.expand_dims(x, axis=1),np.expand_dims(y,axis=1)],axis=1)

            # (u, v) points to a specific direction, with a certain length for the vector
            # 15 can be any other proper number
            offset_num=30
            u = random.randrange(-offset_num,offset_num)
            v = random.randrange(-offset_num,offset_num)
            # number of samples, the naive code contains no strategy to keep all samples uniformly located in the mask
            num_samples = 50
            
            for k in range(num_samples):
                r = random.choice(xy)
                coords_tmp=[]
                for k_frame in range(1,video_length):   
                    coords_tmp.append([r[1],r[0], int(k_frame*u/(video_length-1)), int(k_frame*v/(video_length-1))])         
                coords.append(coords_tmp)
    offset_x=random.randrange(-2,2)
    offset_y=random.randrange(-2,2)

    for k_frame in range(0,video_length-1):
        sparse = np.zeros((1, 2, image.size(2), image.size(3)), dtype=np.float32)
        mask = np.zeros((1, 2, image.size(2), image.size(3)), dtype=np.float32)
        for arr_ in coords:
            arr=arr_[k_frame]
            sparse[0, :, int(arr[1]), int(arr[0])] = np.array(arr[2:4])
            mask[0, :, int(arr[1]), int(arr[0])] = np.array([1, 1])

        sparse = torch.from_numpy(sparse).cuda()
        mask = torch.from_numpy(mask).cuda()
        model.set_input(image, torch.cat([sparse, mask], dim=1), None)

        # inference for flow field
        tensor_dict = model.eval(ret_loss=False)
        flow = tensor_dict['flow_tensors'][0].cpu().numpy().squeeze().transpose(1,2,0).astype(np.float16)
        # clear artifacts out of masks
        tmp = np.sum(seg,axis=2)
        tmp = np.expand_dims((tmp>0), axis=2)
        tmp = np.concatenate([tmp,tmp],axis=2)
        flow = flow*tmp

        flow[:,:,0]+=offset_x*(k_frame+1)
        flow[:,:,1]+=offset_y*(k_frame+1)
    
        flow = torch.from_numpy(flow).unsqueeze(0).permute(0,3,1,2).cuda()
        
        save_image=flow_warping(image, flow)
        save_depth=flow_warping(depth, flow)
        #save image
        def unnormalize(img,mean,std):
            #img is [3,h,w] or [1,h,w]
            img=img.cpu().numpy()
            img=img.transpose(1,2,0)
            img=img*std+mean
            return img
        
        save_image=unnormalize(save_image[0],data_mean,data_div)
        save_depth=unnormalize(save_depth[0],depth_mean,depth_div)
        save_image=np.clip(save_image,0,255).astype(np.uint8)[:,:,::-1]
        save_depth=np.clip(save_depth,0,255).astype(np.uint8)

        cv2.imwrite(os.path.join(videopath,str(k_frame+1)+'.png'),save_image)#rgb 2 bgr
        cv2.imwrite(os.path.join(depthpath,str(k_frame+1)+'.png'),save_depth)
        pass

    saved_num+=1

    print(f'{str(saved_num)} completed {i}| total {len(clean_images)}')

