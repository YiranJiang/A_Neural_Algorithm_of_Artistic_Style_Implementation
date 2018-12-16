A Neural Algorithm of Artistic Style TensorFlow Implementation
==========================

This is the TensorFlow implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)


Here is one of our output image.

![image](fig/2.jpg)

<br>

## File Organization:

```
project/
├── pretrained_model/
├── data/
├── fig/
├── output/
├── main.ipynb
├── train.py
├── utils.py
└── VGG.py
```
<br>

- pretrained_model/: Currently empty, but need to put the VGG-19 model here for reproduction.
- data/: Contains Content & Style input image.
- fig/: Contains figure in readme file.
- output/: The reproduced output image will be saved within this directory.
- main.ipynb: A notebook file which we could use to reproduce the result.
- train.py: Mainly a class which constructs the neural network we need and the training functions.
- utils.py: Mainly stores image processing and visualization functions.
- VGG.py: Contains the part of reading the pretrained VGG-19 model parameters into our style learning model.


<br>

## Reproduction:

Firstly, please download the pretrained VGG-19 model [here](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) and put it in the under **pretrained_model/**.

To reproduce our result, you could run the notebook file **main.ipynb**. There are chunks which reproduce different types of result including the **running time**, **output image** as well as **training loss history**. The animation part comes from my own idea and is also very fancy and worth trying.

<br>

## Contribution statement:

The VGG.py and part of uils.py is implemented by Du Guo(<dg2999@columbia.edu>) and the other part is by me, Yiran Jiang(<yj2462@columbia.edu>). Also thanks to [here](https://github.com/anishathalye/neural-style), where the preprocessing idea comes from. Yu Tong(<yt2594@columbia.edu>) helps us explore by generating output images using different set of parameters.


<br>
<br>

12/15/2018

Yiran Jiang(<yj2462@columbia.edu>)
