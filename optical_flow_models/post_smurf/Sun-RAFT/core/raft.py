import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from update import BasicUpdateBlock
from extractor import BasicEncoder, BasicContextEncoder
from corr import  CorrBlock
from utils.utils import coords_grid
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def downflow(flow, mode='bilinear', factor=0.125):
    old_size = (flow.shape[2], flow.shape[3])
    new_size = (int(factor * flow.shape[2]), int(factor * flow.shape[3]))
    u_scale = new_size[1]/old_size[1]
    v_scale = new_size[0]/old_size[0]
    resized_flow = F.interpolate(flow, size=new_size, mode=mode, align_corners=True) #b 2 h w
    resized_flow_split = torch.split(resized_flow, 1,  dim=1)
    rescaled_flow = torch.cat([u_scale*resized_flow_split[0], v_scale*resized_flow_split[1]], dim=1)
    
    return rescaled_flow

class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args["corr_levels"] = 4
        args["corr_radius"] = 4

        if 'dropout' not in self.args:
            self.args["dropout"] = 0

        if 'alternate_corr' not in self.args:
            self.args["alternate_corr"] = False

        self.fnet = BasicEncoder(output_dim=256, norm_fn=args["fnet_norm"]) 
        self.cnet = BasicContextEncoder(output_dim=hdim+cdim, norm_fn=args["cnet_norm"]) 

        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, scale=8)
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def initialize_flow8(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def initialize_flow16(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//16, W//16).to(img.device)
        coords1 = coords_grid(N, H//16, W//16).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def get_grid(self, img, scale):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//scale, W//scale).to(img.device)
        return coords0

    def upsample_flow(self, flow, mask, scale=8):
        """ Upsample flow field [H/scale, W/scale, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(scale * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, scale*H, scale*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False, bw=True):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        batchsize, channels, h, w = image1.shape
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args["mixed_precision"]):
            
            fmap1, fmap2 = self.fnet([image1, image2], bw=bw) 
            
            # run the context network
            if bw:
                images_fw_bw = torch.cat((image1, image2), dim=0)
                cnet, cnet_4, cnet_2 = self.cnet(images_fw_bw) 
            else:
                cnet, cnet_4, cnet_2 = self.cnet(image1)
            
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        if bw:
            coords0, coords1 = self.initialize_flow8(images_fw_bw)
            flow_bw_predictions = []
        else:
            coords0, coords1 = self.initialize_flow8(image1)

        flow_predictions = []
        
        if flow_init is not None:
                coords1 = coords1 + flow_init

        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
       
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args["corr_radius"])    

      
        for itr in range(iters):
            coords1 = coords1.detach()
            
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.args["mixed_precision"]):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            # upsample predictions
            flow_up = self.upsample_flow(coords1 - coords0, up_mask, scale=8)
  
            if bw:
                flow_up_fw, flow_up_bw = flow_up.split([batchsize, batchsize], dim=0)
                flow_bw_predictions.append(flow_up_bw)
                flow_predictions.append(flow_up_fw)
            else:
                flow_predictions.append(flow_up)

        if test_mode:
                return coords1 - coords0, flow_up
        
        if bw:
            return flow_predictions, flow_bw_predictions
        else:
            return flow_predictions
