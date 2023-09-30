#!/usr/bin/python
from __future__ import print_function

### python lib
import os, sys, argparse, glob, re, math, copy, pickle
from datetime import datetime
import numpy as np

### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

### custom lib
#from networks.resample2d_package.modules.resample2d import Resample2d
from pwc import Resample2d
import networks
import datasets
import utils
from loguru import logger



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Adaptive Video Temporal Consistency")

    ### model options
    parser.add_argument('-model',           type=str,     default="TransformNet",   help='TransformNet') 
    parser.add_argument('-nf',              type=int,     default=32,               help='#Channels in conv layer') #default 32
    parser.add_argument('-blocks',          type=int,     default=5,                help='#ResBlocks') 
    parser.add_argument('-norm',            type=str,     default='IN',             choices=["BN", "IN", "none"],   help='normalization layer')
    parser.add_argument('-model_name',      type=str,     default='DepthTC',           help='path to save model')

    ### dataset options
    parser.add_argument('-data_dir',        type=str,     default='data',           help='path to data folder')
    parser.add_argument('-list_dir',        type=str,     default='lists',          help='path to lists folder')
    parser.add_argument('-checkpoint_dir',  type=str,     default='checkpoints',    help='path to checkpoint folder')
    parser.add_argument('-crop_size',       type=int,     default=192,              help='patch size')  #default 192
    parser.add_argument('-geometry_aug',    type=int,     default=1,                help='geometry augmentation (rotation, scaling, flipping)')
    parser.add_argument('-order_aug',       type=int,     default=1,                help='temporal ordering augmentation')
    parser.add_argument('-scale_min',       type=float,   default=0.5,              help='min scaling factor')
    parser.add_argument('-scale_max',       type=float,   default=2.0,              help='max scaling factor')
    parser.add_argument('-sample_frames',   type=int,     default=5,               help='#frames for training') #default 11
    
    ### loss options
    parser.add_argument('-alpha',           type=float,   default=50,             help='alpha for computing visibility mask')
    parser.add_argument('-loss',            type=str,     default="L1",             help="optimizer [Options: SGD, ADAM]")
    parser.add_argument('-w_ST',            type=float,   default=100,              help='weight for short-term temporal loss')
    parser.add_argument('-w_LT',            type=float,   default=100,              help='weight for long-term temporal loss')
    parser.add_argument('-w_PP',           type=float,   default=100,                help='weight for ping-pang loss')
    parser.add_argument('-w_MMN',            type=float,    default=5,                help='weight for MMN loss')
    ### training options
    parser.add_argument('-solver',          type=str,     default="ADAM",           choices=["SGD", "ADAIM"],   help="optimizer")
    parser.add_argument('-momentum',        type=float,   default=0.9,              help='momentum for SGD')
    parser.add_argument('-beta1',           type=float,   default=0.9,              help='beta1 for ADAM')
    parser.add_argument('-beta2',           type=float,   default=0.999,            help='beta2 for ADAM')
    parser.add_argument('-weight_decay',    type=float,   default=0,                help='weight decay')
    parser.add_argument('-batch_size',      type=int,     default=8,                help='training batch size') #default 4
    parser.add_argument('-train_epoch_size',type=int,     default=2000,             help='train epoch size') #default 1000
    parser.add_argument('-valid_epoch_size',type=int,     default=100,              help='valid epoch size')
    parser.add_argument('-epoch_max',       type=int,     default=100 ,             help='max #epochs') #default:100


    ### learning rate options
    parser.add_argument('-lr_init',         type=float,   default=1e-4,             help='initial learning Rate')  #default 1e-4
    parser.add_argument('-lr_offset',       type=int,     default=20,               help='epoch to start learning rate drop [-1 = no drop]') #default 20
    parser.add_argument('-lr_step',         type=int,     default=20,               help='step size (epoch) to drop learning rate')
    parser.add_argument('-lr_drop',         type=float,   default=0.5,              help='learning rate drop ratio')  #default 0.5
    parser.add_argument('-lr_min_m',        type=float,   default=0.1,              help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')
    

    ### other options
    parser.add_argument('-seed',            type=int,     default=20020129,             help='random seed to use')
    parser.add_argument('-threads',         type=int,     default=8,                help='number of threads for data loader to use')
    parser.add_argument('-suffix',          type=str,     default='',               help='name suffix')
    parser.add_argument('-gpu',             type=int,     default=0,                help='gpu device id')
    parser.add_argument('-cpu',             action='store_true',                    help='use cpu?')
    
    opts = parser.parse_args()

    ### adjust options
    opts.cuda = (opts.cpu != True)
    opts.lr_min = opts.lr_init * opts.lr_min_m
    
    ### default model name
    if opts.model_name == 'none':
        
        opts.model_name = "%s_B%d_nf%d_%s" %(opts.model, opts.blocks, opts.nf, opts.norm)

    if opts.suffix != "":
        opts.model_name += "_%s" %opts.suffix


    opts.size_multiplier = 2 ** 6 ## Inputs to FlowNet need to be divided by 64
    
    print(opts)


    torch.manual_seed(opts.seed)
    if opts.cuda:
        torch.cuda.manual_seed(opts.seed)


    ### model saving directory
    opts.model_dir = os.path.join(opts.checkpoint_dir, opts.model_name)
    print("========================================================")
    print("===> Save model to %s" %opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    
    ### initialize model
    print('===> Initializing model from %s...' %opts.model)
    model = networks.__dict__[opts.model](opts, nc_in=2, nc_out=1)
    model.train()
    ### initialize optimizer
    if opts.solver == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opts.lr_init, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.solver == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr_init, weight_decay=opts.weight_decay, betas=(opts.beta1, opts.beta2))
    else:
        raise Exception("Not supported solver (%s)" %opts.solver)
    

    ### resume latest model
    name_list = glob.glob(os.path.join(opts.model_dir, "model_epoch_*.pth"))
    epoch_st = 0
    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]


    if epoch_st > 0:

        print('=====================================================================')
        print('===> Resuming model from epoch %d' %epoch_st)
        print('=====================================================================')

        ### resume latest model and solver
        model, optimizer = utils.load_model(model, optimizer, opts, epoch_st)
        #logger.debug(f"optimizer:{optimizer.param_groups}")
    else:
        ### save epoch 0
        utils.save_model(model, optimizer, opts)


    print(model)

    num_params = utils.count_network_parameters(model)

    print('\n=====================================================================')
    print("===> Model has %d parameters" %num_params)
    print('=====================================================================')


    ### initialize loss writer
    loss_dir = os.path.join(opts.model_dir, 'loss')
    loss_writer = SummaryWriter(loss_dir)
    
    
    ### load pwcnet instead flownet
    import pwc
    FlowNet=pwc.Flownet(requires_grad=False)
    print("===> Load pwc-net")
    
    ### convert to GPU
    device = torch.device("cuda" if opts.cuda else "cpu")

    model = model.to(device)
    FlowNet = FlowNet.to(device)
    
    model.train()

    
    ### create dataset
    train_dataset = datasets.MultiFramesDataset(opts, "train")

    
    ### start training
    while model.epoch < opts.epoch_max:

        model.epoch += 1

        ### re-generate train data loader for every epoch
        data_loader = utils.create_data_loader(train_dataset, opts, "train")

        ### update learning rate
        current_lr = utils.learning_rate_decay(opts, model.epoch)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        ## submodule
        flow_warping = Resample2d().to(device)
        downsampler = nn.AvgPool2d((2, 2), stride=2).to(device)


        ### criterion and loss recorder
        if opts.loss == 'L2':
            criterion = nn.MSELoss(size_average=True)
        elif opts.loss == 'L1':
            criterion = nn.L1Loss(size_average=True)
        else:
            raise Exception("Unsupported criterion %s" %opts.loss)
        
        
        ### start epoch
        ts = datetime.now()
        for iteration, batch in enumerate(data_loader, 1):
            
            total_iter = (model.epoch - 1) * opts.train_epoch_size + iteration

            ### convert data to cuda
            frame_ori = []
            frame_d = []
            frame_gt=[]
            for t in range(opts.sample_frames*2-1):
                frame_ori.append(batch[t * 3].to(device))
                frame_d.append(batch[t * 3 + 1].to(device))
                frame_gt.append(batch[t * 3 + 2].to(device))
            frame_o = []
            frame_o.append(frame_d[0]) ## first frame

            ### get batch time
            data_time = datetime.now() - ts

            ts = datetime.now()

            optimizer.zero_grad()

            lstm_state = None
            ST_loss = 0
            LT_loss = 0
            PP_loss= 0
            MMN_loss=0
            ### forward
            for t in range(1, 2*opts.sample_frames-1):

                frame_ori1 = frame_ori[t - 1]
                frame_ori2 = frame_ori[t]
                frame_d2 = frame_d[t]

                if t == 1:
                    frame_o1 = frame_o[t - 1]
                else: 
                    frame_o1 = frame_o2.detach()    ## previous output frame

                frame_o1.requires_grad = False 

                ### model input   important      
                inputs = torch.cat((frame_d2, frame_o1), dim=1)
                ### forward model
                output, lstm_state = model(inputs, lstm_state)
                ### residual learning
                frame_o2 = output + frame_d2

                ## detach from graph and avoid memory accumulation
                lstm_state = utils.repackage_hidden(lstm_state)

                frame_o.append(frame_o2)


                ### short-term temporal loss
                if opts.w_ST > 0:
                    ### compute flow (from I2 to I1)
                    flow_i21 = FlowNet(frame_ori2, frame_ori1)
                    ### warp I1 and O1
                    warp_i1 = flow_warping(frame_ori1, flow_i21)
                    warp_o1 = flow_warping(frame_o1, flow_i21)

                    ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
                    noc_mask2 = torch.exp( -opts.alpha * torch.sum(frame_ori2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)
                    
                    ST_loss += opts.w_ST * criterion(frame_o2 * noc_mask2, warp_o1 * noc_mask2)
                    
                

            ## end of forward


            ### long-term temporal loss
            if opts.w_LT > 0:

                t1 = 0
                for t2 in range(t1 + 2, 2*opts.sample_frames-1):

                    frame_ori1 = frame_ori[t1]
                    frame_ori2 = frame_ori[t2]

                    frame_o1 = frame_o[t1].detach() ## make a new Variable to avoid backwarding gradient
                    frame_o1.requires_grad = False

                    frame_o2 = frame_o[t2]

                    ### compute flow (from I2 to I1)
                    flow_i21 = FlowNet(frame_ori2, frame_ori1)
                    
                    ### warp I1 and O1
                    warp_i1 = flow_warping(frame_ori1, flow_i21)
                    warp_o1 = flow_warping(frame_o1, flow_i21)

                    ### compute non-occlusion mask: exp(-alpha * || F_i2 - Warp(F_i1) ||^2 )
                    noc_mask2 = torch.exp( -opts.alpha * torch.sum(frame_ori2 - warp_i1, dim=1).pow(2) ).unsqueeze(1)

                    LT_loss += opts.w_LT * criterion(frame_o2 * noc_mask2, warp_o1 * noc_mask2)

                ### end of t2
            ### end of w_LT
            ### ping-pang loss
            if opts.w_PP > 0:
                for t in range(2*opts.sample_frames-1):
                    frame_d1=frame_o[t]
                    frame_d2=frame_o[-1-t]
                    PP_loss+=opts.w_PP*criterion(frame_d1,frame_d2)

            ### end of ping-pang loss

            ###MMN loss
            if opts.w_MMN > 0:
                for t in range(2*opts.sample_frames-1):
                    depth=frame_d[t]
                    gt=frame_gt[t]
                    MMN_loss+=opts.w_MMN*criterion(depth,gt)



            ### overall loss
        
            overall_loss = ST_loss + LT_loss + PP_loss + MMN_loss

            # logger.debug(f"loss_grad:{overall_loss.requires_grad}")
            
            ### backward loss
            overall_loss.backward()
            ### update parameters
            optimizer.step()
            
            network_time = datetime.now() - ts


            ### print training info
            info = "[GPU %d]: " %(opts.gpu)
            info += "Epoch %d; Batch %d / %d; " %(model.epoch, iteration, len(data_loader))
            info += "lr = %s; " %(str(current_lr))

            ## number of samples per second
            batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
            info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" %(data_time.total_seconds(), network_time.total_seconds(), batch_freq)
            
            info += "\tmodel = %s\n" %opts.model_name

            ### print and record loss
            if opts.w_ST > 0:
                loss_writer.add_scalar('ST_loss', ST_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("ST_loss", ST_loss.item())

            if opts.w_LT > 0:
                loss_writer.add_scalar('LT_loss', LT_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("LT_loss", LT_loss.item())
            if opts.w_PP > 0:
                loss_writer.add_scalar('PP_loss', PP_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("PP_loss", PP_loss.item())
            if opts.w_MMN > 0:
                loss_writer.add_scalar('MMN_loss', MMN_loss.item(), total_iter)
                info += "\t\t%25s = %f\n" %("MMN_loss", MMN_loss.item())

            loss_writer.add_scalar('Overall_loss', overall_loss.item(), total_iter)
            info += "\t\t%25s = %f\n" %("Overall_loss", overall_loss.item())

            print(info)

        ### end of epoch

        ### save model
        utils.save_model(model, optimizer, opts)


            
