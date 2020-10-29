#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 18:49:20 2020

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

if __name__=='__main__':
    WeightsPath='./ScaleFactor4TrainIteration60.hdf5'
    ImageShape=(200,400)
    ScaleFactor=4
    epochs=200
    SRCNN=SRCNN.SRCNN(image_shape=(32,32,1),scale_factor=ScaleFactor)
    SRCNN_model=SRCNN.Load_SRCNN(WeightsPath)
    patch_size=(32,32)
    slidingDis=2
    ImageShape=(200,400)
    ScaleFactor=4
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
    
    print(PSNR(HR_Image,LR_Image))
    print(PSNR(HR_Image,PredictData))