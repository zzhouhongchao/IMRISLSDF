import os
import glob
import sys
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data




class DatasetLPBA(Data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
       
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]  
        
        Start = self.files[index].rfind('/S')
        LSDFDir = 'data/LPBA/LSDF' + self.files[index][Start:-7]
        
        ImgCSF = sitk.GetArrayFromImage(sitk.ReadImage(LSDFDir + '_seg_CSF.nii.gz'))[np.newaxis, ...] 
        ImgGM = sitk.GetArrayFromImage(sitk.ReadImage(LSDFDir + '_seg_GM.nii.gz'))[np.newaxis, ...]
        ImgWM = sitk.GetArrayFromImage(sitk.ReadImage(LSDFDir + '_seg_WM.nii.gz'))[np.newaxis, ...]

       
        return img_arr,ImgCSF,ImgGM,ImgWM,self.files[index]
    
