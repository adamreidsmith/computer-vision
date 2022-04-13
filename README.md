# Computer Vision Models
This repository contains a few computer vision models employed using [PyTorch](https://pytorch.org) and [MediaPipe](https://google.github.io/mediapipe/).

# Description of Models

## MediaPipe Models
The [face detection](/Face%20Detection), [face mesh](/Face%20Mesh), [hand tracking](/Hand%20Tracking), [pose estimation](/Pose%20Estimation), and [gesture volume control](/Gesture%20Volume%20Control) models all utilize the MediaPipe machine learning framework which facilitates implementation of fast, customizable ML pipelines.  The former four are modules to simplify the use of the corresponding MediaPipe solution and provide examples of their use, while the latter uses the hand tracking module to allow the user to control their devices volume via hand gestures.

## Deblurring Models

### Deblur GAN
The [Image Deblurring](/Image%20Deblurring) directory contains a few implementations of image deblurring deep convolutional GAN's created using the PyTorch machine learning framework.  In [`deblur_gan.py`](/Image%20Deblurring/deblur_gan.py), the utilize a [residual learning](https://arxiv.org/pdf/1512.03385.pdf) architecture.  Residual networks have proven to be quite effective in computer vision tasks as they are easier to optimize and can gain accuracy from considerably increased depth compared to traditional deep convolutional networks.  The generator is composed of 9 ResNet blocks, each consisting of two convolutional operations with batch normalization and ReLU activation functions, and a single shortcut connection.  The ResNet blocks are preceded and followed by several layers of downsampling and upsampling operations, respectively.  The discriminator follows a more standard deep convolutional network architecture, utilizing batch normalization and leaky ReLU activation functions.  The generator's loss function is a combination of the adversarial loss determined by the discriminator's performance, and a VGG loss.  VGG loss uses the pre-trained convolutional neural network VGG16 used by ImageNet for object recognition.  The intermediate layers in this network provide feature activations for the input image, and the feature activations at the same layer for the target and generated image can be compared using L1 loss to produce _feature losses_.  These feature losses ensure the generator is generating images that closely resemble the blurred input images and not drastically changing features.  Combining the feature losses with the adversarial loss from the discriminator train the generator to output sharp images that also strongly resemble the blurred input image.

### Deblur Wasserstein GAN
[`deblur_gan_wasserstein.py`](/Image%20Deblurring/deblur_gan_wasserstein.py) alters the deblur GAN described above to fit the [Wasserstein GAN](https://arxiv.org/abs/1701.07875) architecture.  The Wasserstein GAN aims to improve stability when training and provide a loss function that better correlates with the quality of the generated images.  The discriminator changes from a classifier to a critic that scores the _realness_ or _fakeness_ of an image.  Implementation of the Wasserstein GAN only requires a few alterations to the deblur GAN.  Namely, we must
1. use a linear activation function (as opposed to a sigmoid) on the last layer of the discriminator,
2. use labels of -1 for artificial images and 1 for real images as opposed to 0 and 1,
3. use the Wasserstein loss to train the generator and discriminator models,
4. constrain the discriminators weights to a small interval centered at 0 (For instance [-0.01, 0.01].  This enforces 1-Lipschitz continuity of the discriminator, which is required by the loss function to ensure stability and ******makes sure it doesnt grow too much******),
5. update the discriminator model more times than the generator model,
6. and use the [RMSprop version](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html) of gradient descent with a small learning rate and no momentum to train the models.

### Adding Gradient Penalty
[`deblur_gan_wasserstein_gp.py`](/Image%20Deblurring/deblur_gan_wasserstein_gp.py) improves upon the previously described Wasserstein GAN by implementing gradient penalty instead of clipping the weights of the discriminator.  Gradient clipping can lead to problems such as vanishing gradients and poor convergence as it limits the discriminators ability to improve and achieve local optima.  The goal of gradient penalty is to address this by enforcing a constraint such that the gradients of the discriminator’s output with respect to the inputs to have unit norm.  This ensures 1-Lipschitz continuity and does not suffer from the same problems as the original Wasserstein GAN architecture, allowing for easier optimization and convergence.  This is achieved by adding a term to the discriminator's loss function that penalizes the discriminator when its gradient norm is larger than 1.  This technique has shown to be effective in experiments, but is more computationally intensive.

Due to a lack of available computing resources, I was not able to train any of the image deblurring models to a significant degree, so I am unfortunately unable to report results.

# Technologies Used
 * [Python 3](https://www.python.org)
 * [PyTorch](https://pytorch.org)
 * [MediaPipe](https://mediapipe.dev)
 * [OpenCV](https://opencv.org)
