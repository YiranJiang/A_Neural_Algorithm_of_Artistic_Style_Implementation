#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:28:13 2018

@author: Yiran Jiang
"""

import time
from scipy import misc
from matplotlib import pyplot as PLT
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
from utils import *
from VGG import *

class Neural_Style(object):
    def __init__(self,network,rand_seed, input_content_path, input_style_path,content_layers, style_layers,wl):
        # Store input values
        self.network = network
        self.rand_seed = rand_seed
        self.param,self.mean_pixel = get_param(network)
        self.input_content_path = input_content_path
        self.input_style_path = input_style_path
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.wl = wl
    
    def imageProcess(self, image_content_path, image_style_path, scale=1):
        # Preprocess input image
        
        image_content_size = ImageFunction().imageGetSize(image_content_path)
        image_style_size = ImageFunction().imageGetSize(image_style_path)
        
        # Scale the image to reduce processing time. We obtain the minimum height and width from content and style input and final multiply it by a scale constant.
        
        self.height = int(min(image_content_size[1], image_style_size[1])*scale)
        self.width = int(min(image_content_size[0], image_style_size[0])*scale)
        
        # Resize the style & content input to the same size.
        input_content = ImageFunction().imageRead(image_content_path, self.width, self.height, original = False)
        input_style = ImageFunction().imageRead(image_style_path, self.width, self.height, original = False)
        return input_content, input_style
    
    def get_feature(self,layers,pivot_image):
        image = tf.placeholder('float', shape=(1,self.height, self.width,3))
        # Construct the VGG net
        vgg_net = VGG19(self.param,image)
        # The preprocess idea comes from https://github.com/anishathalye/neural-style, but did not use ANY other part of code in it.
        pivot_image = np.array([preprocess(pivot_image,self.mean_pixel)])
        F = {}
        G = {}
        with tf.Session() as sess:
            for i in layers:
                # Calculate F value and G value by forward run the net.
                F[i] = vgg_net[i].eval(feed_dict={image:pivot_image})
                _,_,_,N = F[i].shape
                features = F[i].reshape(-1,N)
                G[i] = np.matmul(features.T, features)

        return F,G

    def train(self,alpha,beta,learning_rate, iterations,scale,init = 'Content'):
        
        input_content, input_style = self.imageProcess(self.input_content_path, self.input_style_path, scale =scale)
        
        # Different initial value choice: 'White' means blank image, 'Random' means randomized image, 'Content' means initial value = Content input, 'Style' means initial value = Style input. Different initial value could have affect the final results. Also, we could watch different animations from our training, which is cool. This part comes from my own idea and is not recorded in the paper.
        
        if init == 'White':
            initial = np.zeros((self.height, self.width,3))
        elif init == 'Random':
            initial = np.random.normal(size =(self.height,self.width,3))*0.255
        elif init == 'Content':
            initial = input_content
        elif init == 'Style':
            initial = input_Style

        initial = np.array([preprocess(initial,self.mean_pixel)])
        blank_image = tf.Variable(initial)
        train_vgg_net = VGG19(self.param,blank_image)


        F,_ = self.get_feature(self.content_layers,input_content)
        _,G = self.get_feature(self.style_layers,input_style)
        
        content_loss = 0
        content_loss_list = []
        style_loss = 0
        style_loss_list = []

        #Calculate Content loss
        for i in self.content_layers:
            content_loss_list.append(self.wl*(0.5*tf.reduce_sum(tf.pow((train_vgg_net[i] - F[i]),2))))
            content_loss = tf.add(content_loss_list[-1],content_loss)

        #Calculate Style loss
        for i in self.style_layers:
            #Calculate weight value
            _, h, w, N = train_vgg_net[i].shape
            M = h*w
            this_weight = 1/(4*int((N*M))**2)
            this_weight = tf.cast(this_weight, dtype=tf.float64)
            
            #Calvulate A value in the paper.
            features = tf.reshape(train_vgg_net[i], (-1, N))
            this_A = tf.matmul(tf.transpose(features), features)
            
            #Calvulate total loss.
            style_loss_list.append(this_weight*self.wl*tf.reduce_sum(tf.pow((this_A - G[i]), 2)))
            style_loss = tf.add(style_loss_list[-1],style_loss)
            loss = alpha*content_loss + beta*style_loss

        with tf.name_scope('train_step'):
            # Use AdamOptimizer to train the result. There are different types of optimizer available, while we tried between Adam & RMSprop(two most stable & popular) and found that Adam works slightly better.
            
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
        with tf.Session() as sess:
            # 'frames' is for animation, 'loss_hist' is for plotting loss.
            frames = []
            loss_hist = []
            
            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            for i in range(iterations):
                if i % 10 == 0:
                    # Append the output image to frame every 10 iterations.
                    this_image = blank_image.eval()
                    this_image = unprocess(this_image, self.mean_pixel)[0]
                    this_image = ImageFunction().imageProcess(this_image)
                    frames.append(this_image)
                
                if i % 50 == 0:
                    # Record the loss every 50 iterations.
                    this_loss = loss.eval()
                    loss_hist.append(this_loss)
                    print('Total Loss: ',this_loss)
                    print('iteration {} starts'.format(i))
                    print("--- %s seconds ---" % (time.time() - start_time))
                train_step.run()

            value_out = blank_image.eval()
            # Unprocess the output image
            value_out = unprocess(value_out, self.mean_pixel)[0]
            value_out = ImageFunction().imageProcess(value_out)
            return(value_out,loss_hist,frames)


