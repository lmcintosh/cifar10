#!/usr/bin/env python
#modify.py
import re
import numpy as np

f = open('cifar10_quick_train_lane01.prototxt')
g = open('cifar10_quick_test_lane01.prototxt')
lines_train = [line.rstrip('\n') for line in f]
lines_test  = [line.rstrip('\n') for line in g]
f.close()
g.close()

pixel_res = 32


def product(list):
    p = 1
    for i in list:
        p *= i
    return p

def getLayers(lines_train):
    '''Returns the layer names in a caffe convolutional neural network.'''
    indicesOfLayers  = [i for (i,line) in enumerate(lines_train) if ('layers' in line)]
    namesOfLayers    = [lines_train[i+1] for i in indicesOfLayers]
    return indicesOfLayers, namesOfLayers



def numLayers(namesOfLayers):
    '''Returns the number of convolutional layers in a caffe convolutional neural network.'''
    convLayers = [(namesOfLayers[i].find('conv') > 0) for i in xrange(len(namesOfLayers))]
    return sum(convLayers)



def numNeurons(lines_train):
    '''Returns number of neurons in a caffe convolutional neural network.
    # neurons = # feature maps * resolution of feature maps 
    e.g. (conv3 layer has 64 feature maps * (8 * 8 subsampled dimension) =  4096 neurons)
    '''

    num_output  = []
    kernel_size = []
    stride      = []

    for (idx,l) in enumerate(lines_train):
        if l.find('num_output') > 0:
            num_output.append(([int(s) for s in l.split() if s.isdigit()])[0])
        if l.find('kernel_size') > 0 and lines_train[idx-1].find('pool') < 0:
            kernel_size.append(([int(s) for s in l.split() if s.isdigit()])[0])
        if l.find('stride') > 0:
            stride.append(([int(s) for s in l.split() if s.isdigit()])[0])

    neurons = []
    for (idx,nFeatureMaps) in enumerate(num_output):
        my_stride = product(stride[0:(2*idx+1)])
        neurons.append(nFeatureMaps * (pixel_res/my_stride) * (pixel_res/my_stride)) 
    
    return sum(neurons)



def numParameters(lines_train):
    '''Returns the number of parameters in a caffe convolutional neural network.
    For a given layer, 
    # params = # feature maps * (kernel depth * kernel height * kernel width) + # feature maps (bias)
    e.g. original conv3 has 64*(32*5*5) + 64 = 51264 parameters.
    '''
    



    return nParams

f = open('param_dir/cifar10_quick_train_lane01-N-K-S.txt')
f.write('\n'.join(translated_stuff))
f.close()
