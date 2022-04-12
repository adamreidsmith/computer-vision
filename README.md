# Computer Vision Models
This repository contains a few computer vision models employed using [PyTorch](https://pytorch.org) and [MediaPipe](https://google.github.io/mediapipe/).

# Description of Models

## MediaPipe Models
The [face detection](/Face%20Detection), [face mesh](/Face%20Mesh), [hand tracking](/Hand%20Tracking), [pose estimation](/Pose%20Estimation), and [gesture volume control](/Gesture%20Volume%20Control) models all utilize the MediaPipe machine learning framework which facilitates implementation of fast, customizable ML pipelines.  The former four are modules to simplify the use of the corresponding MediaPipe solution and provide examples of their use, while the latter uses the hand tracking module to allow the user to control their devices volume via hand gestures.

## Deblurring Models
The [Image Deblurring](/Image%20Deblurring) directory contains a few implementations of image deblurring deep convolutional GAN's created using the PyTorch machine learning framework.  In [`deblur_gan.py`](/Image%20Deblurring/deblur_gan.py), the utilize a [residual learning](https://arxiv.org/pdf/1512.03385.pdf) architecture.  Residual networks have proven to be quite effective in computer vision tasks as they are easier to optimize and can gain accuracy from considerably increased depth compared to traditional deep convolutional networks.  The generator is composed of 9 ResNet blocks, each consisting of two convolutional operations with batch normalization and ReLU activation functions, and a single shortcut connection.  The ResNet blocks are preceded and followed by several layers of downsampling and upsampling operations, respectively.  The discriminator follows a more standard deep convolutional network architecture utilizing batch normalization and leaky ReLU activation functions.
