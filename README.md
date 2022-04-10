# Computer Vision Models
This repository contains a few computer vision models employed using [PyTorch](https://pytorch.org) and [MediaPipe](https://google.github.io/mediapipe/).

# Description of Models

## MediaPipe Models
The [face detection](/Face%20Detection), [face mesh](/Face%20Mesh), [hand tracking](/Hand%20Tracking), [pose estimation](/Pose%20Estimation), and [gesture volume control](/Gesture%20Volume%20Control) models all utilize the MediaPipe machine learning framework which facilitates implementation of fast, customizable ML pipelines.  The former four are modules to simplify the use of the corresponding MediaPipe solution and provide examples of their use, while the latter uses the hand tracking module to allow the user to control their devices volume via hand gestures.

## Deblurring Models
The [Image Deblurring](/Image%20Deblurring) directory contains a few implementations of deep convolutional GAN's for deblurring images created using the PyTorch machine learning framework.
