#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:21:31 2020

@author: nephilim
"""

# SRCNN Reference by Learning a Deep Convolutional Network for Image Super-Resolution

import keras 

class SRCNN():
    def __init__(self,image_shape,scale_factor=4):
        self.__name__='SRCNN'
        self.scale_factor=scale_factor
        self.image_shape=image_shape

    # Building the SRCNN Network
    def Build_SRCNN(self):
        input_image=keras.layers.Input(shape=self.image_shape,name='LowResolutionImage')
        x=keras.layers.Conv2D(filters=64,kernel_size=(9,9),padding='same',activation='relu')(input_image)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=32,kernel_size=(1,1),padding='same',activation='relu')(x)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=1,kernel_size=(5,5),padding='same',activation='sigmoid')(x)
        # x=keras.layers.Dropout(rate=0.2)(x)        
        model=keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='mse')
        return model
    
    def Train_SRCNN(self,model,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name):
        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath=save_path_name+'.h5',monitor='val_loss',save_best_only=True),\
                    keras.callbacks.TensorBoard(log_dir='./TensorBoard',histogram_freq=1,write_graph=True,write_images=True)]
        history=model.fit(inputs_train,outputs_train,epochs=epochs,batch_size=32,callbacks=callbacks_list,validation_data=(inputs_validation,outputs_validation),validation_split=0.1)
        model.save_weights(save_path_name+'.hdf5')
        test_loss=model.evaluate(inputs_validation,outputs_validation)
        return history,test_loss,model
    
    def Load_SRCNN(self,WeightsPath):
        input_image=keras.layers.Input(shape=self.image_shape,name='LowResolutionImage')
        x=keras.layers.Conv2D(filters=64,kernel_size=(9,9),padding='same',activation='relu')(input_image)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=32,kernel_size=(1,1),padding='same',activation='relu')(x)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=1,kernel_size=(5,5),padding='same',activation='sigmoid')(x)
        # x=keras.layers.Dropout(rate=0.2)(x)        
        model=keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='mse')
        model.load_weights(WeightsPath)
        return model
    
class ResidualSRCNN():
    def __init__(self,image_shape,scale_factor=4):
        self.__name__='ResidualSRCNN'
        self.scale_factor=scale_factor
        self.image_shape=image_shape

    # Building the SRCNN Network
    def Build_ResidualSRCNN(self):
        input_image=keras.layers.Input(shape=self.image_shape,name='LowResolutionImage')
        x=keras.layers.Conv2D(filters=64,kernel_size=(9,9),padding='same',activation='relu')(input_image)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=32,kernel_size=(1,1),padding='same',activation='relu')(x)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.add([input_image,x])
        x=keras.layers.Conv2D(filters=1,kernel_size=(5,5),padding='same',activation='sigmoid')(x)
        # x=keras.layers.Dropout(rate=0.2)(x)        
        model=keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='mse')
        return model
    
    def Train_ResidualSRCNN(self,model,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name):
        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath=save_path_name+'.h5',monitor='val_loss',save_best_only=True),\
                    keras.callbacks.TensorBoard(log_dir='./TensorBoard',histogram_freq=1,write_graph=True,write_images=True)]
        history=model.fit(inputs_train,outputs_train,epochs=epochs,batch_size=32,callbacks=callbacks_list,validation_data=(inputs_validation,outputs_validation),validation_split=0.1)
        model.save_weights(save_path_name+'.hdf5')
        test_loss=model.evaluate(inputs_validation,outputs_validation)
        return history,test_loss,model
    
    def Load_ResidualSRCNN(self,WeightsPath):
        input_image=keras.layers.Input(shape=self.image_shape,name='LowResolutionImage')
        x=keras.layers.Conv2D(filters=64,kernel_size=(9,9),padding='same',activation='relu')(input_image)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=32,kernel_size=(1,1),padding='same',activation='relu')(x)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.add([input_image,x])
        x=keras.layers.Conv2D(filters=1,kernel_size=(5,5),padding='same',activation='sigmoid')(x)
        # x=keras.layers.Dropout(rate=0.2)(x)        
        model=keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(learning_rater=0.001),loss='mse')
        model.load_weights(WeightsPath)
        return model
    
    
    
    
    
    
    
    
    
    
    
    