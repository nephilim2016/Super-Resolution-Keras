#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:29:58 2020

@author: nephilim
"""

import SRCNN
import my_Im2col
import DataNormalized
import math
import numpy as np
import skimage.transform
from matplotlib import pyplot,cm
import os 


def PSNR(OriginalImage,BlurredImage):
    mse=np.sum((OriginalImage-BlurredImage)**2)/OriginalImage.size
    PSNR_=10*math.log10(np.max(OriginalImage)**2/mse)
    return PSNR_

def Patch2TrainData(Patch,PatchSize):
    TrainData=np.zeros(((Patch.shape[1],)+PatchSize+(1,)))
    for idx in range(Patch.shape[1]):
        TrainData[idx,:,:,0]=Patch[:,idx].reshape(PatchSize)
    return TrainData

def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def LoadData(FilePath,FileName,ImageShape,ScaleFactor):
    if isinstance(FileName,list):
        HR_Image_List=list()
        LR_Image_List=list()
        for filename in FileName:
            data_temp=np.load(FilePath+'/'+filename)
            data_temp=skimage.transform.resize(data_temp,output_shape=ImageShape,mode='symmetric')
            HR_Image=DataNormalized.DataNormalized(data_temp)
            LR_Image=HR_Image[::ScaleFactor,::ScaleFactor]
            LR_Image=skimage.transform.resize(LR_Image,output_shape=ImageShape,mode='symmetric')
            HR_Image_List.append(HR_Image)
            LR_Image_List.append(LR_Image)
        patch_size=(32,32)
        slidingDis=2

        for idx in range(len(FileName)):
            HR_Image_Patch_,_=GetPatch(HR_Image_List[idx],patch_size,slidingDis)
            LR_Image_Patch_,_=GetPatch(LR_Image_List[idx],patch_size,slidingDis)
            if idx==0:
                LR_Image_Patch=LR_Image_Patch_
                HR_Image_Patch=HR_Image_Patch_
            else:
                LR_Image_Patch=np.hstack((LR_Image_Patch,LR_Image_Patch_))
                HR_Image_Patch=np.hstack((HR_Image_Patch,HR_Image_Patch_))
            
    else:
        data_temp=np.load(FilePath+'/'+filename)
        data_temp=skimage.transform.resize(data_temp,output_shape=ImageShape,mode='symmetric')
        HR_Image=DataNormalized.DataNormalized(data_temp)
        LR_Image=HR_Image[::ScaleFactor,::ScaleFactor]
        LR_Image=skimage.transform.resize(LR_Image,output_shape=ImageShape,mode='symmetric')
        patch_size=(32,32)
        slidingDis=2
        HR_Image_Patch,_=GetPatch(HR_Image,patch_size,slidingDis)
        LR_Image_Patch,_=GetPatch(LR_Image,patch_size,slidingDis)
    TrainData_inputs=Patch2TrainData(LR_Image_Patch,patch_size)
    TrainData_outputs=Patch2TrainData(HR_Image_Patch,patch_size)
    return TrainData_inputs,TrainData_outputs
    
if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    FilePath='./SeismicModel'
    FileName=['bp94Vp.npy','BP_1994_vp.npy','salt_x12.5_z6.25_rho.npy','SEAM_Vp.npy','vel_z6.25m_x12.5m_exact.npy','vel_z6.25m_x12.5m_nosalt.npy','vp_marmousi-ii.npy']
    ImageShape=(200,400)
    ScaleFactor=4
    epochs=200
    SRCNN=SRCNN.SRCNN(image_shape=(32,32,1),scale_factor=ScaleFactor)
    SRCNN_model=SRCNN.Build_SRCNN()
    
    patch_size=(32,32)
    slidingDis=2
    TrainData_inputs,TrainData_outputs=LoadData(FilePath,FileName,ImageShape,ScaleFactor)
    inputs_train=TrainData_inputs
    outputs_train=TrainData_outputs
    inputs_validation=TrainData_inputs[:1000]
    outputs_validation=TrainData_outputs[:1000]
    save_path_name='./ScaleFactor4TrainIteration60'
    history,test_loss,Model=SRCNN.Train_SRCNN(SRCNN_model,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name)
    
    HR_Image=np.load('./SeismicModel/vp_marmousi-ii.npy')
    HR_Image=skimage.transform.resize(HR_Image,output_shape=ImageShape,mode='symmetric')
    HR_Image=DataNormalized.DataNormalized(HR_Image)
    LR_Image=HR_Image[::ScaleFactor,::ScaleFactor]
    LR_Image=skimage.transform.resize(LR_Image,output_shape=ImageShape,mode='symmetric')
    LR_Image_Patch_,Patch_Idx=GetPatch(LR_Image,patch_size,slidingDis)
    TestData=Patch2TrainData(LR_Image_Patch_,patch_size)
    Predict_Patch=SRCNN_model.predict(TestData)
    x_decoded=Predict_Patch[:,:,:,0]
    rows,cols=my_Im2col.ind2sub(np.array(ImageShape)-patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(ImageShape)
    Weight=np.zeros(ImageShape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:]
        IMout[row:row+patch_size[0],col:col+patch_size[1]]+=block
        Weight[row:row+patch_size[0],col:col+patch_size[1]]+=np.ones(patch_size)
        count+=1
    PredictData=IMout/Weight
    