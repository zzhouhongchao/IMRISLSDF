import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def conv_block(dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

class ResizeTransformer_block(nn.Module):
    
    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x

class RegstrationEncoder(nn.Module):
    def __init__(self, dim, enc_nf):
        super(RegstrationEncoder, self).__init__()
        self.enc_nf = enc_nf  
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 3, 2, batchnorm=None))
    
    def forward(self, src, tgt):  
        x = torch.cat([src, tgt], dim=1)   
        x_enc = [x]
        for i, l in enumerate(self.enc):  
            x = l(x_enc[-1])
            x_enc.append(x)  
        return x_enc
    
class LSDFEncoder(nn.Module):
    def __init__(self, dim, enc_nf):
        super(LSDFEncoder, self).__init__()
        self.enc_nf = enc_nf  # 编码
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 1 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 3, 2, batchnorm=None))

    def forward(self, src):  
        x_enc = [src]
        for i,l in enumerate(self.enc):  
            x = l(x_enc[-1])    
            x_enc.append(x)   
        return x_enc 

class LSDFDecoder(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(LSDFDecoder, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
       
        self.LSDFdec = nn.ModuleList()
        self.LSDFdec.append(conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.LSDFdec.append(conv_block(dim, dec_nf[0] * 3, dec_nf[1], batchnorm=bn))  # 2
        self.LSDFdec.append(conv_block(dim, dec_nf[1] * 3, dec_nf[2], batchnorm=bn))  # 3
        self.LSDFdec.append(conv_block(dim, dec_nf[2] * 2, dec_nf[3], batchnorm=bn))  # 4 
        self.LSDFdec.append(conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.LSDFdec.append(conv_block(dim, dec_nf[4] + 3, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.LSDF_vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.LSDF = conv_fn(dec_nf[-1], 3, kernel_size=1,stride=1)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self,LSDFenc,Renc):
        
        y = LSDFenc[-1]    
        for i in range(3):
            y = self.LSDFdec[i](y) 
            y = self.upsample(y)   
            y = torch.cat([y,LSDFenc[-(i+2)],Renc[-(i+2)]],dim=1)   
        y = self.LSDFdec[3](y)   
        y = self.LSDFdec[4](y)   
        if self.full_size:
            y = self.upsample(y)    
        y = torch.cat([y, LSDFenc[0],Renc[0]], dim=1)   
        y = self.LSDFdec[5](y)     
        if self.vm2:
            y = self.LSDF_vm2_conv(y)    
        LSDF = self.LSDF(y)      
        CSFLSDF = LSDF[:,0:1,:,:,:]
        GMLSDF = LSDF[:,1:2,:,:,:]
        WMLSDF = LSDF[:,2:3,:,:,:]
        
        return CSFLSDF,GMLSDF,WMLSDF
      
class RegistrationFirstDecoder(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(RegistrationFirstDecoder, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        self.RegistrationDec1 = nn.ModuleList()
        self.RegistrationDec1.append(conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.RegistrationDec1.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.RegistrationDec1.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3
        self.RegistrationDec1.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.RegistrationDec1.append(conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5
        
        if self.full_size:
            self.RegistrationDec1.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.Registration_vm2_conv1 = conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.FirstFlow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)
        
    def forward(self,Renc):
        y =Renc[-1]
        for i in range(3):
            y = self.RegistrationDec1[i](y)  
            y = self.upsample(y)  
            y = torch.cat([y,Renc[-(i+2)]],dim=1)  
        y = self.RegistrationDec1[3](y)  
        y = self.RegistrationDec1[4](y)  
        if self.full_size:
            y = self.upsample(y)     
        y = torch.cat([y, Renc[0]], dim=1) 
        y = self.RegistrationDec1[5](y)  
        if self.vm2:
            y = self.Registration_vm2_conv1(y)     
        FirstFlow = self.FirstFlow(y)  
        
        return  FirstFlow
    
class RegistrationSecondDecoder(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(RegistrationSecondDecoder, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        self.RegistrationDec2 = nn.ModuleList()
        self.RegistrationDec2.append(conv_block(dim, 67, dec_nf[0], batchnorm=bn))  # 1
        self.RegistrationDec2.append(conv_block(dim, 96, dec_nf[1], batchnorm=bn))  # 2
        self.RegistrationDec2.append(conv_block(dim, 96, dec_nf[2], batchnorm=bn))  # 3
        self.RegistrationDec2.append(conv_block(dim, 64, dec_nf[3], batchnorm=bn))  # 4
        self.RegistrationDec2.append(conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5
        
        if self.full_size:
            self.RegistrationDec2.append(conv_block(dim, dec_nf[4] + 3, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.Registration_vm2_conv2 = conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.SecondFlow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self,Renc,Firstflowdown4,Lenc):
        y = torch.cat([Renc[-1],Firstflowdown4,Lenc[-1]],dim=1)
        y = self.RegistrationDec2[0](y)
        y = self.upsample(y)
        y = torch.cat([y,Renc[-2],Lenc[-2]],dim=1)    
        
        y = self.RegistrationDec2[1](y)   
        y = self.upsample(y)    
        y = torch.cat([y,Renc[-3],Lenc[-3]],dim=1)
        
        y = self.RegistrationDec2[2](y)  
        y = self.upsample(y)    
        y = torch.cat([y,Renc[-4],Lenc[-4]],dim=1)
        
        y = self.RegistrationDec2[3](y)  
        y = self.RegistrationDec2[4](y)  
        if self.full_size:
            y = self.upsample(y)    
        y = torch.cat([y, Renc[0],Lenc[0]], dim=1)  
        y = self.RegistrationDec2[5](y)  
        if self.vm2:
            y = self.Registration_vm2_conv2(y)    
        SecondFlow = self.SecondFlow(y)    
        if self.bn:
            SecondFlow = self.batch_norm(SecondFlow)
        
        return SecondFlow
                


class RegistrationNet(nn.Module):
    def __init__(self, dim, vol_size,enc_nf, dec_nf, bn=None, full_size=True):
        super(RegistrationNet, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        self.Encoder = RegstrationEncoder(len(vol_size), enc_nf)
        self.FirstDecoder = RegistrationFirstDecoder(len(vol_size), enc_nf,dec_nf)
        self.SecondDecoder = RegistrationSecondDecoder(len(vol_size), enc_nf,dec_nf)
        self.LSDFencoder = LSDFEncoder(len(vol_size), enc_nf)
        self.LSDFdecoder = LSDFDecoder(len(vol_size), enc_nf,dec_nf)
        
        self.STN = SpatialTransformer(vol_size)
        self.down1 = conv_block(dim, 3, dec_nf[-1], 3, 2, batchnorm=None)
        self.down2 = conv_block(dim, dec_nf[-1], dec_nf[-2], 3, 2, batchnorm=None)
        self.down3 = conv_block(dim, dec_nf[-2], dec_nf[-1], 3, 2, batchnorm=None)
        self.down4 = conv_block(dim, dec_nf[-1], 3, 3, 2, batchnorm=None)
           
    def forward(self,moving,fixed):
        Renc = self.Encoder(moving,fixed)
        FirstFlow = self.FirstDecoder(Renc)
        FirstWarp = self.STN(moving,FirstFlow)
        
        movingenc = self.LSDFencoder(moving)
        movingCSFLSDF,movingGMLSDF,movingWMLSDF = self.LSDFdecoder(movingenc,Renc)
        
        fixedenc = self.LSDFencoder(fixed)
        fixedCSFLSDF,fixedGMLSDF,fixedWMLSDF = self.LSDFdecoder(fixedenc,Renc)
        
        firstwarpenc = self.LSDFencoder(FirstWarp)
        firstwarpCSFLSDF,firstwarpGMLSDF,firstwarpWMLSDF = self.LSDFdecoder(firstwarpenc,Renc)
        
        Firstflowdown1 = self.down1(FirstFlow)
        Firstflowdown2 = self.down2(Firstflowdown1)
        Firstflowdown3 = self.down3(Firstflowdown2)
        Firstflowdown4 = self.down4(Firstflowdown3)
        

        SecondFlow = self.SecondDecoder(Renc,Firstflowdown4,firstwarpenc)
        
        Flow = self.STN(FirstFlow,SecondFlow)
        Flow = torch.add(SecondFlow,Flow)
        
        return FirstFlow,FirstWarp,SecondFlow,Flow,movingCSFLSDF,movingGMLSDF,movingWMLSDF,fixedCSFLSDF,fixedGMLSDF,fixedWMLSDF,firstwarpCSFLSDF,firstwarpGMLSDF,firstwarpWMLSDF
        
        
        
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]  
        grids = torch.meshgrid(vectors) 
       
        grid = torch.stack(grids)  
        grid = torch.unsqueeze(grid, 0) 
        grid = grid.type(torch.FloatTensor) 
        self.register_buffer('grid', grid)  

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow  
        shape = flow.shape[2:]

       
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
             
        if len(shape) == 2:
          
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
          
            new_locs = new_locs.permute(0, 2, 3, 4, 1)  
            
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)
  