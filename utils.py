#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:28:13 2018

@author: Du Guo, Yiran Jiang
"""

from scipy import misc
from matplotlib import pyplot as PLT
import matplotlib.animation as animation
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=100):
    PLT.close()  # or else nbagg sometimes plots in the previous cell
    fig = PLT.figure()
    patch = PLT.imshow(frames[0])
    PLT.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)
# These two functions come from Machine Learning text book: https://proquest.safaribooksonline.com/book/programming/9781491962282/firstchapter#X2ludGVybmFsX0h0bWxWaWV3P3htbGlkPTk3ODE0OTE5NjIyODIlMkZpZG0xNDAwMjYwMzg5MjYzNTJfaHRtbCZxdWVyeT0=


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


class ImageFunction(object):
    def imageGetSize(self, image_path):
        img = Image.open(image_path)
        return img.size
    
    def imageRead(self, image_path, height = None, width = None, original = True):
        if type(image_path) != str:
            raise TypeError('image_path should be a path(str) of your picture.')
        img = Image.open(image_path)
        if original == True:
            img_resized = np.zeros((img.size[1], img.size[0], 3))
            img_resized[:,:,:] = img
        else:
            if height is None or width is None:
                raise ValueError('You need to input height and width to resize your picture.')
            img = img.resize((height, width))
            img_empty = np.zeros((width, height,3))
            img_empty[:,:,:] = img
            img_resized = img_empty
        return img_resized

    def imageShow(self, image):
        image = np.clip(image, 0, 255).astype('uint8')
        PLT.figure(figsize=(5,4))
        PLT.imshow(image)
        PLT.axis('off')
        PLT.show()
    
    def imageProcess(self,image):
        # Process image by clipping and type transferring
        image = np.clip(image, 0, 255).astype('uint8')
        return(image)

