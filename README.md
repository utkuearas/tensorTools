# Tensor Tools

## What is Tensor Tools ?

Tensor Tools is a library which contain many ready popular methods that used in machine learning and computer vision.

## Which methods that include ?

Tensor Tools contain:
* MobileNets Bottleneck Blocks
  * MobileNetV1 Block
  * MobileNetv2 Block
  * Relu 6 ( Number can be changed )
* SqueezeNet Blocks
  * Fire Module
* ResNet Blocks
  * Residual Block
  
## What is advantage of using tensorTools library ?

* Building models quickly
* Easy To Use
* Flexibility ( You can change some parameters in ready methods such as kernel size )

# Documentation

### tensorTools.mobileNetv1Block(x , depthWiseKernel = (3,3) , filters = None , useStride = False , strides = (2,2))

x ( Mandatory ) => Must be a tensor with at least 4d dimension (None , 32 , 32 , 3)

depthWiseKernel ( Arbitary ) => Default value is same with original article 

filters ( Arbitary ) => Must be an integer , default value equal to double of x's channel

useStride ( Arbitary ) => If True , stride will apply to Depthwise Convolution layer

strides ( Arbitary ) => If useStride is False , this is unnecessary. If useStride is True, stride value can be changeable

### tensorTools.mobileNetv2Block(x , outputChannel , depthWiseKernel = (3,3) , t = 6 , useStride = True , strides = (2,2))

x ( Mandatory ) => Must be a tensor with at least 4d dimension (None , 32 , 32 , 3)

outputChannel ( Mandatory ) => Must be an integer , specify the output channel size

depthWiseKernel ( Arbitary ) => Default value is same with original article

filters ( Arbitary ) => Must be an integer , default value equal to double of x's channel

t ( Arbitary ) => t used for calculate first expand convolution layer (expand = x's channel * t) , default value equal to 6

useStride ( Arbitary ) => If True , stride will apply to Depthwise Convolution layer

strides ( Arbitary ) => If useStride is False , this is unnecessary. If useStride is True, stride value can be changeable
