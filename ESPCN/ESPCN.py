#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:55:45 2020

@author: nephilim
"""

# ESPCN Reference by Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
import keras
import tensorflow as tf

class ESPCN():
    def __init__(self,image_shape,scale_factor=4):
        self.__name__='ESPCN'
        self.image_shape=image_shape
        self.scale_factor=scale_factor
    
    def Build_ESPCN(self):
        input_image=keras.layers.Input(shape=self.image_shape,name='LowResolutionImage')
        x=keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='tanh')(input_image)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='tanh')(x)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=self.scale_factor**2,kernel_size=(3,3),padding='same',activation='sigmoid')(x)
        x=tf.nn.depth_to_space(x,self.scale_factor)
        model=keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='mse')
        return model
        
    def Train_ESPCN(self,model,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name):
        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath=save_path_name+'.h5',monitor='val_loss',save_best_only=True),\
                    keras.callbacks.TensorBoard(log_dir='./TensorBoard',histogram_freq=1,write_graph=True,write_images=True)]
        history=model.fit(inputs_train,outputs_train,epochs=epochs,batch_size=32,callbacks=callbacks_list,validation_data=(inputs_validation,outputs_validation),validation_split=0.1)
        model.save_weights(save_path_name+'.hdf5')
        test_loss=model.evaluate(inputs_validation,outputs_validation)
        return history,test_loss,model
    
    def Load_ESPCN(self,WeightsPath):
        input_image=keras.layers.Input(shape=self.image_shape,name='LowResolutionImage')
        x=keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='tanh')(input_image)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='tanh')(x)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=self.scale_factor**2,kernel_size=(3,3),padding='same',activation='sigmoid')(x)
        x=tf.nn.depth_to_space(x,self.scale_factor)
        model=keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(learning_rater=0.0001),loss='mse')
        model.load_weights(WeightsPath)
        return model
    
class ResidualESPCN():
    def __init__(self,image_shape,scale_factor=4):
        self.__name__='ResidualESPCN'
        self.image_shape=image_shape
        self.scale_factor=scale_factor
    
    def Build_ResidualESPCN(self):
        input_image=keras.layers.Input(shape=self.image_shape,name='LowResolutionImage')
        x=keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='tanh')(input_image)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='tanh')(x)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.add([input_image,x])
        x=keras.layers.Conv2D(filters=self.scale_factor**2,kernel_size=(3,3),padding='same',activation='sigmoid')(x)
        x=tf.nn.depth_to_space(x,self.scale_factor)
        model=keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),loss='mse')
        return model
        
    def Train_ResidualESPCN(self,model,epochs,inputs_train,outputs_train,inputs_validation,outputs_validation,save_path_name):
        callbacks_list=[keras.callbacks.ModelCheckpoint(filepath=save_path_name+'.h5',monitor='val_loss',save_best_only=True),\
                    keras.callbacks.TensorBoard(log_dir='./TensorBoard',histogram_freq=1,write_graph=True,write_images=True)]
        history=model.fit(inputs_train,outputs_train,epochs=epochs,batch_size=32,callbacks=callbacks_list,validation_data=(inputs_validation,outputs_validation),validation_split=0.1)
        model.save_weights(save_path_name+'.hdf5')
        test_loss=model.evaluate(inputs_validation,outputs_validation)
        return history,test_loss,model
    
    def Load_ResidualESPCN(self,WeightsPath):
        input_image=keras.layers.Input(shape=self.image_shape,name='LowResolutionImage')
        x=keras.layers.Conv2D(filters=64,kernel_size=(5,5),padding='same',activation='tanh')(input_image)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='tanh')(x)
        x=keras.layers.Dropout(rate=0.2)(x)
        x=keras.layers.add([input_image,x])
        x=keras.layers.Conv2D(filters=self.scale_factor**2,kernel_size=(3,3),padding='same',activation='sigmoid')(x)
        x=tf.nn.depth_to_space(x,self.scale_factor)
        model=keras.models.Model(inputs=input_image,outputs=x)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(learning_rater=0.0001),loss='mse')
        model.load_weights(WeightsPath)
        return model
    
    