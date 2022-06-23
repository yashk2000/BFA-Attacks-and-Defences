# Design and Analysis of Novel Bit-flip Attacks and Defense Strategies for DNNs

_In this paper, we present novel bit-flip attack (BFA) algorithms for DNNs, along with techniques for defending against the attack. Our attack algorithms leverage information about the layer importance, such that a layer is considered important if it has high-ranked feature maps. We first present a classwise- targeted attack that degrades the accuracy of just one class in the dataset. Comparative evaluation with related works shows the effectiveness of our attack algorithm. We finally propose multiple novel defense strategies against untargeted BFAs. We comprehensively evaluate the robustness of both large-scale CNNs (VGG19, ResNext50, AlexNet and ResNet) and compact CNNs (MobileNet-v2, ShuffleNet, GoogleNet and SqueezeNet) towards BFAs. We also reveal a valuable insight that compact CNNs are highly vulnerable to not only well-crafted BFAs such as ours, but even random BFAs. Also, defense strategies are less effective on compact CNNs. This fact makes them unsuitable for use in security-critical domains._

## Code implementation

This repository has code implementations of the following parts from our paper:
- [LI-T-BFA](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/attacks/litbfa)
- [Random BFA](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/attacks/random)
- [Odd-even quantization](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/defences/quantization)
  - [S1*](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/defences/quantization/s1_quantzation.py)
  - [S2*](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/defences/quantization/s2_quantzation.py) 
- [Changing from max pooling to average pooling](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/defences/pooling) - Contains model definitions with updated pooling layers
- [Changing activation function from ReLU to ReLU6 or tanh](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/defences/activations) - Contains model definitations with updated activation functions.  
- [Freezing crucial layers](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/defences/layer_freeze)
- [Performing Hrank](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/hrank)

## HRank

To prevent a user from having to run the HRank code everytime to obtain layer importance, we have provided layers rankings for all models in [json format](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/hrank_jsons), containing the layer name and respective ranking in `float` format:
- [Alexnet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/hrank_jsons/alexnet.json)
- [Resnet18](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/hrank_jsons/resnet18.json)
- [Resnext50](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/hrank_jsons/resnext50.json)
- [VGG19](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/hrank_jsons/vgg19.json)
- [Mobilenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/hrank_jsons/mobilenet.json)
- [Googlenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/hrank_jsons/googlenet.json)
- [Squeezenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/hrank_jsons/squeezenet.json)
- [Shufflenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/hrank_jsons/shufflenet.json)

## Model Definitions

Additionally, we have added [model definitions](https://github.com/yashk2000/BFA-Attacks-and-Defences/tree/main/models) for the quantized along with the code to convert a model from the PyTorch Hub into the qunatized format:
- [Quantization code](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/quantization.py)
- [Alexnet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/alexnet_imagenet.py)
- [Densenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/densenet_imagenet.py)
- [VGG](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/vgg_imagenet.py)
- [Resnet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/resnet_imagenet.py)
- [Mobilenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/mobilenet_imagenet.py)
- [Googlenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/googlenet_imagenet.py)
- [Shufflenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/shufflenet_imagenet.py)
- [Squeezenet](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/squeezenet_imagenet.py)
- [Resnet modified for cifar10 dataset](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/models/resnet_cifar.py)

Using these models:

```python

from models import alexnet_imagenet, densenet_imagenet, googlenet_imagenet, resnet_imagenet, shufflenet_imagenet, squeezenet_imagenet, vgg_imagenet, resnet_cifar, vanilla_resnet_cifar, mobilenet_imagenet

pretrained = True # set true to download a pretrained model and quantize it, false otherwise
num_classes = 1000 # number of classes in the dataset, 1000 for imagenet, 10 for cifar 

if args.arch == 'alexnet':
    net = alexnet_imagenet.alexnet_quan(pretrained, num_classes)
elif args.arch == 'googlenet':
    net = googlenet_imagenet.googlenet_quan(pretrained, True, num_classes)
elif args.arch == 'mobilenet':
    net = mobilenet_imagenet.mobilenet_v2_quan(pretrained, True, num_classes)
elif args.arch == 'resnet18':
    net = resnet_imagenet.resnet18_quan(pretrained, True, num_classes)
elif args.arch == 'resnext50':
    net = resnet_imagenet.resnext50_32x4d_quan(pretrained, True, num_classes)
elif args.arch == 'shufflenet':
    net = shufflenet_imagenet.shufflenet_v2_x2_0(pretrained, True, num_classes)
elif args.arch == 'squeezenet':
    net = squeezenet_imagenet.squeezenet1_1(pretrained, True, num_classes)
elif args.arch == 'vgg19':
    net = vgg_imagenet.vgg19(pretrained, True, num_classes)
```

**A simple way to use all the code is by modifying this [driver class](https://github.com/yashk2000/BFA-Attacks-and-Defences/blob/main/main.py). This will be updated soon with instructions on how to run code through this. The file contains implementations for all dataloaders, model loading, initializaing attacks, training models, loading a model from a checkpoint, logging all outputs, etc.**

If you use this code, cite the following paper:

```
@inproceedings { ref151,
title            = "Design and Analysis of Novel Bit-flip Attacks and Defense Strategies for DNNs",
year             = "2022",
author           = "Yash Khare and Kumud Lakara and Maruthi S Inukonda and Sparsh Mittal and Mahesh Chandra and Arvind Kaushik",
booktitle        = "5th IEEE Conference on Dependable and Secure Computing (DSC)",
address          = "Edinburgh, UK",
publisher        = "IEEE",
}
```
