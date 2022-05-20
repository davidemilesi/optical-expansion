from __future__ import print_function
import sys
import cv2
import pdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
from utils.io import mkdir_p
from utils.util_flow import write_flow, save_pfm
from utils.flowlib import point_vec, warp_flow

import matplotlib.pyplot as plt
from pathlib import Path
import struct

cudnn.benchmark = False

parser = argparse.ArgumentParser(description='VCN+expansion')
parser.add_argument('--dataset', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/ssd/kitti_scene/training/',
                    help='dataset path')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--outdir', default='output',
                    help='output dir')
parser.add_argument('--testres', type=float, default=1,
                    help='resolution')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarse cost volume size')
args = parser.parse_args()


# dataloader
if args.dataset == '2015':
    from dataloader import kitti15list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015val':
    from dataloader import kitti15list_val as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015vallidar':
    from dataloader import kitti15list_val_lidar as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015test':
    from dataloader import kitti15list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'seq':
    from dataloader import seqlist as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sinteltest':
    from dataloader import sintellist as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sintel':
    from dataloader import sintellist_val as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  

max_h = int(maxh // 64 * 64)
max_w = int(maxw // 64 * 64)
if max_h < maxh: max_h += 64
if max_w < maxw: max_w += 64
maxh = max_h
maxw = max_w


mean_L = [[0.33,0.33,0.33]]
mean_R = [[0.33,0.33,0.33]]

# construct model, VCN-expansion
from models.VCN_exp import VCN
model = VCN([1, maxw, maxh], md=[int(4*(args.maxdisp/256)),4,4,4,4], fac=args.fac, 
  exp_unc=('robust' in args.loadmodel))  # expansion uncertainty only in the new model
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('dry run')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


mkdir_p('%s/%s/'% (args.outdir, args.dataset))
def main():
    model.eval()
    ttime_all = []
    for inx in range(len(test_left_img)):
        print(test_left_img[inx])
        imgL_o = cv2.imread(test_left_img[inx])[:,:,::-1]
        imgR_o = cv2.imread(test_right_img[inx])[:,:,::-1]

        # for gray input images
        if len(imgL_o.shape) == 2:
            imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
            imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))

        # resize
        maxh = imgL_o.shape[0]*args.testres
        maxw = imgL_o.shape[1]*args.testres
        max_h = int(maxh // 64 * 64)
        max_w = int(maxw // 64 * 64)
        if max_h < maxh: max_h += 64
        if max_w < maxw: max_w += 64

        input_size = imgL_o.shape
        imgL = cv2.resize(imgL_o,(max_w, max_h))
        imgR = cv2.resize(imgR_o,(max_w, max_h))

        # flip channel, subtract mean
        imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
        imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
        imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
        imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

        # modify module according to inputs
        from models.VCN_exp import WarpModule, flow_reg
        for i in range(len(model.module.reg_modules)):
            model.module.reg_modules[i] = flow_reg([1,max_w//(2**(6-i)), max_h//(2**(6-i))], 
                            ent=getattr(model.module, 'flow_reg%d'%2**(6-i)).ent,\
                            maxdisp=getattr(model.module, 'flow_reg%d'%2**(6-i)).md,\
                            fac=getattr(model.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
        for i in range(len(model.module.warp_modules)):
            model.module.warp_modules[i] = WarpModule([1,max_w//(2**(6-i)), max_h//(2**(6-i))]).cuda()

        # forward
        imgL = Variable(torch.FloatTensor(imgL).cuda())
        imgR = Variable(torch.FloatTensor(imgR).cuda())
        with torch.no_grad():
            imgLR = torch.cat([imgL,imgR],0)
            model.eval()
            torch.cuda.synchronize()
            start_time = time.time()
            rts = model(imgLR)
            torch.cuda.synchronize()
            ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
            ttime_all.append(ttime)
            flow, occ, logmid, logexp = rts

        # upsampling
        occ = cv2.resize(occ.data.cpu().numpy(),  (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        logexp = cv2.resize(logexp.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        logmid = cv2.resize(logmid.cpu().numpy(), (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
        flow = torch.squeeze(flow).data.cpu().numpy()
        flow = np.concatenate( [cv2.resize(flow[0],(input_size[1],input_size[0]))[:,:,np.newaxis],
                                cv2.resize(flow[1],(input_size[1],input_size[0]))[:,:,np.newaxis]],-1)
        flow[:,:,0] *= imgL_o.shape[1] / max_w
        flow[:,:,1] *= imgL_o.shape[0] / max_h
        flow = np.concatenate( (flow, np.ones([flow.shape[0],flow.shape[1],1])),-1)


        # Convertitore 
        img_dim = (4.70,4.70)
        
        def read_pfm(filename):
            with Path(filename).open('rb') as pfm_file:
                line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
                assert line1 in ('PF', 'Pf')

                channels = 3 if "PF" in line1 else 1
                width, height = (int(s) for s in line2.split())
                scale_endianess = float(line3)
                bigendian = scale_endianess > 0
                scale = abs(scale_endianess)

                buffer = pfm_file.read()
                samples = width * height * channels
                assert len(buffer) == samples * 4

                fmt = f'{"<>"[bigendian]}{samples}f'
                decoded = struct.unpack(fmt, buffer)
                shape = (height, width, 3) if channels == 3 else (height, width)
                return np.flipud(np.reshape(decoded, shape)) * scale
        
        
        # save predictions
        idxname = test_left_img[inx].split('/')[-1]
        with open('%s/%s/flo-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]),'w') as f:
            #print('percorso')
            #print('%s/%s/flo-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]))
            save_pfm(f,flow[::-1].astype(np.float32))
            
            txt = str('%s/%s/flo-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]))
            x = txt.replace("pfm", "jpg")
            
            fig = plt.figure(figsize=img_dim)
            image = read_pfm(txt)
            plt.imshow(image)
            plt.tight_layout()
            plt.savefig(x)
            plt.close()
            
        flowvis = point_vec(imgL_o, flow)
        cv2.imwrite('%s/%s/visflo-%s.jpg'% (args.outdir, args.dataset,idxname),flowvis)
        imwarped = warp_flow(imgR_o, flow[:,:,:2])
        cv2.imwrite('%s/%s/warp-%s.jpg'% (args.outdir, args.dataset,idxname),imwarped[:,:,::-1])
        with open('%s/%s/occ-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]),'w') as f:
            save_pfm(f,occ[::-1].astype(np.float32))
                        
            txt = str('%s/%s/occ-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]))
            x = txt.replace("pfm", "jpg")
            
            fig = plt.figure(figsize=img_dim)
            image = read_pfm(txt)
            plt.imshow(image)
            plt.tight_layout()
            plt.savefig(x)
            plt.close()
            
        with open('%s/%s/exp-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]),'w') as f:
            save_pfm(f,logexp[::-1].astype(np.float32))
            
            txt = str('%s/%s/exp-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]))
            x = txt.replace("pfm", "jpg")
            
            fig = plt.figure(figsize=img_dim)
            image = read_pfm(txt)
            plt.imshow(image)
            plt.tight_layout()
            plt.savefig(x)
            plt.close()
            
        with open('%s/%s/mid-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]),'w') as f:
            save_pfm(f,logmid[::-1].astype(np.float32))
            
            txt = str('%s/%s/mid-%s.pfm'% (args.outdir, args.dataset,idxname.split('.')[0]))
            x = txt.replace("pfm", "jpg")
            
            fig = plt.figure(figsize=img_dim)
            image = read_pfm(txt)
            plt.imshow(image)
            plt.tight_layout()
            plt.savefig(x)
            plt.close()
            
        torch.cuda.empty_cache()
    print(np.mean(ttime_all))
                
            

if __name__ == '__main__':
    main()

