#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:45:29 2020

@author: nephilim
"""
import numpy as np
import my_Im2col
from matplotlib import pyplot,cm
import skimage.transform
import cv2
import DataNormalized


def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def GetPatchData(ProfileTarget,patch_size,slidingDis):
    Patch,Patch_Idx=GetPatch(ProfileTarget,patch_size,slidingDis)
    data=np.zeros((Patch.shape[1],patch_size[0],patch_size[1]))
    for idx in range(Patch.shape[1]):
        data[idx,:,:]=Patch[:,idx].reshape(patch_size)
    return data

def DisplayPatch(Patch,PatchNoise,numRows,numCols):
    bb=2
    SizeForEachImage=34
    I=np.ones((SizeForEachImage*numRows+bb,SizeForEachImage*numCols+bb))*(-1)
    INoise=np.ones((SizeForEachImage*numRows+bb,SizeForEachImage*numCols+bb))*(-1)
    maxRandom=Patch.shape[0]
    index=np.random.randint(0,maxRandom,size=maxRandom)
    counter=0
    for j in range(numRows):
        for i in range(numCols):
            I[bb+j*SizeForEachImage:(j+1)*SizeForEachImage+bb-2,bb+i*SizeForEachImage:(i+1)*SizeForEachImage+bb-2]=Patch[index[counter]]
            INoise[bb+j*SizeForEachImage:(j+1)*SizeForEachImage+bb-2,bb+i*SizeForEachImage:(i+1)*SizeForEachImage+bb-2]=PatchNoise[index[counter]]
            counter+=1
    # I-=np.min(I)
    # I/=np.max(I)
    # INoise-=np.min(INoise)
    # INoise/=np.max(INoise)
    return I,INoise
    
    

if __name__=='__main__': 
    Image=cv2.cvtColor(cv2.imread('./test/butterfly.bmp'),cv2.COLOR_BGR2GRAY).astype('float')
    Image=skimage.transform.resize(Image,output_shape=(512,512),mode='symmetric')
    Image_=DataNormalized.DataNormalized(Image)
    ImageBlurred=skimage.transform.resize(Image_[0:None:4,0:None:4],output_shape=(512,512),mode='edge')
    patch_size=(32,32)
    slidingDis=4
    ImageBlurred_Patch=GetPatchData(ImageBlurred,patch_size,slidingDis)
    Image_Patch=GetPatchData(Image_,patch_size,slidingDis)
    
    I,INoise=DisplayPatch(Image_Patch,ImageBlurred_Patch,16,16)
    pyplot.figure()
    pyplot.imshow(I,vmax=np.max(Image_Patch),vmin=np.min(Image_Patch),cmap=cm.gray)
    pyplot.axis('off')
    pyplot.savefig('ESPCNPatchImage.png',dpi=1000)
    pyplot.figure()
    pyplot.imshow(INoise,vmax=np.max(Image_Patch),vmin=np.min(Image_Patch),cmap=cm.gray)
    pyplot.axis('off')
    pyplot.savefig('ESPCNPatchNoise.png',dpi=1000)