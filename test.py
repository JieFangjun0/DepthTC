#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, pickle, cv2, time
import numpy as np

### torch lib
import torch
import torch.nn as nn

### custom lib
#from networks.resample2d_package.modules.resample2d import Resample2d
from pwc import Resample2d
import networks
import utils


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Depth Video Temporal Consistency')

    ### model options
    parser.add_argument('-method',          type=str,     required=True,            help='test model name')
    parser.add_argument('-epoch',           type=int,     required=True,            help='epoch')

    ### dataset options
    parser.add_argument('-phase',           type=str,     default="test",           choices=["train", "test"])
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to list folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-redo',            action="store_true",                    help='Re-generate results')

    ### other options
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    
    opts = parser.parse_args()
    opts.cuda = True

    opts.size_multiplier = 2 ** 2 ## Inputs to TransformNet need to be divided by 4

    print(opts)

    if opts.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without -cuda")


    ### load model opts
    opts_filename = os.path.join(opts.checkpoint_dir, opts.method, "opts.pth")
    print("Load %s" %opts_filename)
    with open(opts_filename, 'rb') as f:
        model_opts = pickle.load(f)


    ### initialize model
    print('===> Initializing model from %s...' %model_opts.model)
    model = networks.__dict__[model_opts.model](model_opts, nc_in=2, nc_out=1)


    ### load trained model
    model_filename = os.path.join(opts.checkpoint_dir, opts.method, "model_epoch_%d.pth" %opts.epoch)
    print("Load %s" %model_filename)
    state_dict = torch.load(model_filename)
    model.load_state_dict(state_dict['model'])

    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")
    model = model.to(device)

    model.eval()


    ### load video list
    video_list=os.listdir(os.path.join(opts.data_dir, opts.phase, "MIDAS"))
    video_list.sort()
    times = []

    ### start testing
    for v in range(len(video_list)):

        video = video_list[v]

        print("Test video %d/%d: %s" %( v + 1, len(video_list), video))

        ## setup path
        input_dir = os.path.join(opts.data_dir, opts.phase, "MIDAS", video)
        GT_dir = os.path.join(opts.data_dir, opts.phase,"GT" , video)
        output_dir = os.path.join(opts.data_dir, opts.phase, "output",opts.method, "epoch_%d" %opts.epoch, video)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            

        frame_list = glob.glob(os.path.join(input_dir, "*.png"))
        output_list = glob.glob(os.path.join(output_dir, "*.png"))

        if len(frame_list) == len(output_list) and not opts.redo:
            print("Output frames exist, skip...")
            continue


        ## frame 0
        frame_p1 = utils.read_img(os.path.join(input_dir, "0-dpt_swin2_large_384.png"),grayscale=True)
        output_filename = os.path.join(output_dir, "0.png")
        utils.save_img(frame_p1, output_filename)

        lstm_state = None

        for t in range(1, len(frame_list)):
                
            ### load frames
            frame_o1 = utils.read_img(os.path.join(output_dir, f"{str(t-1)}.png"),grayscale=True)
            frame_p2 = utils.read_img(os.path.join(input_dir, f"{str(t)}-dpt_swin2_large_384.png"),grayscale=True)
            
            ### resize image
            H_orig = frame_p2.shape[0]
            W_orig = frame_p2.shape[1]

            H_sc = int(math.ceil(float(H_orig) / opts.size_multiplier) * opts.size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / opts.size_multiplier) * opts.size_multiplier)
                
            frame_o1 = cv2.resize(frame_o1, (W_sc, H_sc))
            frame_p2 = cv2.resize(frame_p2, (W_sc, H_sc))
            frame_o1=np.expand_dims(frame_o1,2)
            frame_p2=np.expand_dims(frame_p2,2)
            
            with torch.no_grad():

                ### convert to tensor
                frame_o1 = utils.img2tensor(frame_o1).to(device)
                frame_p2 = utils.img2tensor(frame_p2).to(device)
                
                ### model input
                inputs = torch.cat((frame_p2, frame_o1), dim=1)
                
                ### forward
                ts = time.time()
                
                output, lstm_state = model(inputs, lstm_state)
                frame_o2 = frame_p2 + output

                te = time.time()
                times.append(te - ts)

                ## create new variable to detach from graph and avoid memory accumulation
                lstm_state = utils.repackage_hidden(lstm_state) 


            ### convert to numpy array
            frame_o2 = utils.tensor2img(frame_o2)
            
            ### resize to original size
            frame_o2 = cv2.resize(frame_o2, (W_orig, H_orig))
            
            ### save output frame
            output_filename = os.path.join(output_dir, f"{str(t)}.png")
            utils.save_img(frame_o2, output_filename)
                    
        ## end of frame
        utils.make_video(f'{input_dir}','%d-dpt_swin2_large_384.png',os.path.join(output_dir,f'input_video.mp4'),fps=2)
        utils.make_video(f'{GT_dir}','%d.png',os.path.join(output_dir,f'GT_video.mp4'),fps=2)
        utils.make_video(f'{output_dir}','%d.png',os.path.join(output_dir,f'TC_video.mp4'),fps=2) 
    ## end of video
    
    if len(times) > 0:
        time_avg = sum(times) / len(times)
        print("Average time = %f seconds (Total %d frames)" %(time_avg, len(times)))
