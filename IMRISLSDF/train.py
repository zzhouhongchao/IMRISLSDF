# python imports
import os
import glob
import warnings
import sys
import torch
import numpy as np
import SimpleITK as sitk
from torch.optim import Adam
import torch.utils.data as Data
import argparse
import torch.nn.functional as F
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from Model import losses
from Model.datagenerators import DatasetLPBA
from Model.network import RegistrationNet,SpatialTransformer

def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.Log_dir):
        os.makedirs(args.Log_dir)

def train(args):
    make_dirs()
    writer = SummaryWriter(args.Log_dir)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # 日志文件
    log_name = 'LPBA40-IMRISLSDF'
    print("log_name: ", log_name)
    f = open(os.path.join(args.Log_dir, log_name + ".txt"), "a")  

    f_img = sitk.ReadImage("data/LPBA/fixed.nii.gz")
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]   
    
    vol_size = input_fixed.shape[2:] 
    input_fixed = np.repeat(input_fixed, args.batch_size, axis=0) 
    input_fixed = torch.from_numpy(input_fixed).to(device).float()   
     
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]
    
    Registrationnet = RegistrationNet(len(vol_size),vol_size,nf_enc, nf_dec).to(device)
    STN = SpatialTransformer(vol_size).to(device)
    Registrationnet.train()
    STN.train()
    opt = Adam(Registrationnet.parameters(), lr=args.lr)
    criterion_pixelwise = torch.nn.SmoothL1Loss()
    
    train_files = glob.glob(os.path.join(args.train_dir,'train', '*.nii.gz'))
    DS = DatasetLPBA(files=train_files) 
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    for i in range (1,args.epoch + 1 ):
        for data in DL:
            input_moving,ImgCSF,ImgGM,ImgWM,name = data
   
            input_moving = input_moving.to(device).float()
            
            ImgCSF = ImgCSF.to(device).float()
            ImgGM = ImgGM.to(device).float()
            ImgWM = ImgWM.to(device).float()
            

            FirstFlow,FirstWarp,SecondFlow,Flow,movingCSFLSDF,movingGMLSDF,movingWMLSDF,fixedCSFLSDF,fixedGMLSDF,fixedWMLSDF,firstwarpCSFLSDF,firstwarpGMLSDF,firstwarpWMLSDF = Registrationnet(input_moving,input_fixed)
            Secondwarp = STN(FirstWarp,SecondFlow)
            warp = STN(input_moving,Flow)
            
            LSDFwarpCSF = STN(movingCSFLSDF,Flow)
            LSDFwarpGM = STN(movingGMLSDF,Flow)
            LSDFwarpWM = STN(movingWMLSDF,Flow)
            
            
            sim_loss1 = losses.ncc_loss1(FirstWarp,input_fixed)
            sim_loss2 = losses.ncc_loss1(Secondwarp,input_fixed)
            sim_loss = losses.ncc_loss1(warp,input_fixed)
            
            grad_loss1 = losses.gradient_loss(FirstFlow)
            grad_loss2 = losses.gradient_loss(SecondFlow)
            grad_loss = losses.gradient_loss(Flow)
            
            
            csfloss_R = criterion_pixelwise(fixedCSFLSDF,LSDFwarpCSF)
            gmloss_R = criterion_pixelwise(fixedGMLSDF,LSDFwarpGM)
            wmloss_R = criterion_pixelwise(fixedWMLSDF,LSDFwarpWM)
            
            
            NJ_loss1 = losses.NJ_loss()(FirstFlow.permute(0, 2, 3, 4, 1))
            NJ_loss2 = losses.NJ_loss()(SecondFlow.permute(0, 2, 3, 4, 1))
            NJ_loss = losses.NJ_loss()(Flow.permute(0, 2, 3, 4, 1))
            # lsdf loss
            csfloss_M = criterion_pixelwise(movingCSFLSDF,ImgCSF)
            gmloss_M = criterion_pixelwise(movingGMLSDF,ImgGM)
            wmloss_M = criterion_pixelwise(movingWMLSDF,ImgWM)
            
            
            simloss = sim_loss1 + sim_loss2 + sim_loss
            gradloss = grad_loss1 + grad_loss2 + grad_loss
            NJloss = 0.000001 * NJ_loss1 + 0.0000005 * NJ_loss2 + 0.000002 * NJ_loss
            lsdfloss_R = 0.1 * csfloss_R + 0.1* gmloss_R + 0.1 * wmloss_R
            lsdfloss_M =  csfloss_M + gmloss_M + wmloss_M
            
            
            loss = simloss + gradloss + NJloss + lsdfloss_R + lsdfloss_M 
            print("i: %d name: %s loss: %f sim: %f grad: %f NJ: %f lsdfR: %f lsdfM: %f "
                  % (i, name, loss.item(),simloss.item(), gradloss.item(),NJloss.item(),lsdfloss_R.item(),lsdfloss_M.item()),flush=True)
            print("%d,%s,%f,%f,%f,%f,%f,%f"
                  % (i,name, loss.item(),simloss.item(), gradloss.item(),NJloss.item(),lsdfloss_R.item(),lsdfloss_M.item()),file=f)
            
                                                                                                                    
            opt.zero_grad() 
            loss.backward() 
            opt.step()
            
        if (i % 10 == 0):
            writer.add_scalar("loss",loss,i)
            writer.add_scalar("sim_loss1",sim_loss1,i)
            writer.add_scalar("sim_loss2",sim_loss2,i)
            writer.add_scalar("sim_loss",sim_loss,i)
            writer.add_scalar("l2_loss1",grad_loss1,i)
            writer.add_scalar("l2_loss2",grad_loss2,i)
            writer.add_scalar("l2_loss",grad_loss,i)
            writer.add_scalar("csfloss_R",csfloss_R,i)
            writer.add_scalar("gmloss_R",gmloss_R,i)
            writer.add_scalar("wmloss_R",wmloss_R,i)
            writer.add_scalar("csfloss_M",csfloss_M,i)
            writer.add_scalar("gmloss_M",gmloss_M,i)
            writer.add_scalar("wmloss_M",wmloss_M,i)
            writer.add_scalar("NJ_loss1",NJ_loss1,i)
            writer.add_scalar("NJ_loss2",NJ_loss2,i)
            writer.add_scalar("NJ_loss",NJ_loss,i)
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(Registrationnet.state_dict(), save_file_name)
    f.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='1')
    parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-4)
    parser.add_argument("--epoch", type=int, help="number of iterations",
                    dest="epoch", default=800)
    parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
    parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1.0)
    parser.add_argument("--train_dir", type=str, help="data folder with training vols",
                    dest="train_dir", default="data/LPBA/")
    parser.add_argument("--model_dir", type=str, help="data folder with training vols",
                    dest="model_dir", default="model_result/IMRISLSDF/LPBA40")
    parser.add_argument("--Log_dir", type=str, help="data folder with training vols",
                    dest="Log_dir", default="Log/IMRISLSDF/LPBA40")
    args = parser.parse_args()
    train(args)
