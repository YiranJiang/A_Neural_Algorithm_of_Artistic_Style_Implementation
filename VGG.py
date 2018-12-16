#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:28:13 2018

@author: Du Guo
"""

from scipy.io import loadmat
import tensorflow as tf
import numpy as np

def get_param(net):
    """
    :param net: (str) net is a path to matlab file which saves the vgg19 model.
    :return: (dict)(np.array) weights of each layers(except last six layers) and mean of pixel.
    """
    # First, load vgg model(.mat file) downloaded from website:http://www.vlfeat.org/matconvnet/pretrained/
    data = loadmat(net)
    # Get the mean pixel from model
    mean_pixel = data['meta']['normalization'][0][0][0][0][2][0][0]
    # Since we only consider the weight in each leayer, we extract the weights of vgg19 model
    data = data['layers'][0]
    # Save the weights of each layers(except last six layers) in a dictionary, which further be used for feedforward
    # Attension: Because this is the structure of matlab, 
    # in matlab, weights of conv layer are [width, height, in_channels, out_channels]
    # however, in tensorflow, weights of conv layer are [height, width, in_channels, out_channels]
    # so we need to transform the weights and bias to the format in tensorflow
    #'imagenet-vgg-verydeep-19.mat'
    net = {}
    for i in range(len(data) - 6):
        layer_type = data[i][0][0][1][0]
        if layer_type == 'conv':
            layer_name = data[i][0][0][0][0]
            # transform kernel weights
            kernel_weight = data[i][0][0][2][0][0]
            kernel_weight = np.transpose(kernel_weight, (1,0,2,3))
            # transform bias weights
            bias_weight = data[i][0][0][2][0][1]
            bias_weight = bias_weight.reshape(-1)
            net[layer_name] = [kernel_weight, bias_weight]
    return net,mean_pixel

# Now we should build a feedforward vgg19 net using weights of each layer(except the last seven layers)
# See more details about the layers:https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
)

def VGG19(net, image, pool_type = 'AVG'):
    """
    :param net: (dict) weights of each layer of VGG19, that is the first result of get_param function.
    :param image: (np.array) a picture has been read.
    :param pool_type: (str) the type of pooling layer, 'AVG' means mean pooling, 'MAX' means max pooling.
    :return: (dict) output of each layer(except last six layers) given an image.
    """
    feed_forward_layer_output = {}
    current = image
#    current = np.expand_dims(current, axis = 0)
    for idx, name in enumerate(VGG19_LAYERS):
        layer_name = name[:4]
        if layer_name == 'conv':
            kernel_res = tf.nn.conv2d(current, net[name][0], strides=(1,1,1,1), padding='SAME')
            current = tf.nn.bias_add(kernel_res, net[name][1])
        elif layer_name == 'relu':
            current = tf.nn.relu(current)  
        else:
            if pool_type == 'AVG':
                current = tf.nn.avg_pool(current, ksize = (1,2,2,1), strides = (1,2,2,1), padding='SAME')
            elif pool_type == 'MAX':
                current = tf.nn.max_pool(current, kszie = (1,2,2,1), strides = (1,2,2,1),  padding='SAME')
            else:
                raise TypeError("the input of pool_type should be 'AVG' or 'MAX' ")
        feed_forward_layer_output[name] = current
    return feed_forward_layer_output


if __name__ == '__main__':
    from scipy import misc
    from matplotlib import pyplot as PLT
    arr = misc.imread('Content.jpg') 
    PLT.imshow(arr)
    PLT.show()
    net = get_param('imagenet-vgg-verydeep-19.mat')[0]
    arr = np.expand_dims(arr.astype('float32'), axis= 0)
    dic = VGG19(net, arr.astype('float32'))
    with tf.Session() as sess:
        tf.global_variables_initializer()
        aa= sess.run(dic)
        print(len(dic))

        
        
