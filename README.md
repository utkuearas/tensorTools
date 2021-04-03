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

### tensorTools.mobileNetV1Block (x , depthWiseKernel = (3,3) , filters = None , useStride = False , strides = (2,2))

<br>
**Customizable MobileNet V1 Block**
 
 
**x( Mandatory )** => Must be a tensor with at least 4 dimension Example: (None , 32 , 32 , 3)

**depthWiseKernel ( Arbitary )** => Default value is same with original article 

**filters ( Arbitary )** => Must be an integer , default value equal to double of x's channel

**useStride ( Arbitary )** => If True , stride will apply to Depthwise Convolution layer

**strides ( Arbitary )** => If useStride is False , this is unnecessary. If useStride is True, stride value can be changeable


### tensorTools.mobileNetV2Block (x , outputChannel , depthWiseKernel = (3,3) , t = 6 , useStride = True , strides = (2,2))

**Customizable MobileNet V2 Bottleneck Block**

**x ( Mandatory )** => Must be a tensor with at least 4 dimension Example: (None , 32 , 32 , 3)

**outputChannel ( Mandatory )** => Must be an integer , specify the output channel size

**depthWiseKernel ( Arbitary )** => Default value is same with original article

**filters ( Arbitary )** => Must be an integer , default value equal to double of x's channel

**t ( Arbitary )** => t used for calculate first expand convolution layer (expand = x's channel * t) , default value equal to 6

**useStride ( Arbitary )** => If True , stride will apply to Depthwise Convolution layer

**strides ( Arbitary )** => If useStride is False , this is unnecessary. If useStride is True, stride value can be changeable


### tensorTools.ReluN ( n , name = None )


**ReluN is made for mobileNet Relu6 Activation Layer , but you can change the number with this method**


**n ( Mandatory )** => If you want to use default Relu6 Activation, send 6 to parameter n , else you can specify n value

**name ( Arbitary )** => You can specify layer name


### tensorTools.fireModule ( x , squeeze = 16 , expand = 64 , kernels = ( (1,1) , (3,3) ) , bypass = False , complexBypass = False )


**Customizable SqueezeNet Fire Module**


**x ( Mandatory )** => Must be a tensor with at least 4 dimension Example: (None , 32 , 32 , 3)

**squeeze ( Arbitary )** => Must be an integer , specify the filter amount of squeeze layer

**expand ( Arbitary )** => Must be an integer , specify the filter amount of expand layer

**kernels ( Arbitary )** => Must be an list/tuple/array with 2 length , specify the kernels of squeeze and expand layer, default value is same with the article

**bypass ( Arbitary )** => default is False , apply fire module with bypass method

**complexBypass ( Arbitary )** => default is False , apply fire module with complex bypass method


### tensorTools.residualBlock ( x , blockType = 'type2' , filters = None , filKerSelection = 'auto' , kernels = None , useStrideFirst = False , strides = (2,2))


**Customizable ResNet Residual Block**


**x ( Mandatory )** => Must be a tensor with at least 4 dimension Example: (None , 32 , 32 , 3)

**blockType ( Arbitary )** => Options: type1/type2/custom, 'type1' is residual block with 2 convolutional layer , 'type2' is residual block with 3 convolutional layer , if blockType is 'custom' , both kernels and filters must be given (length of kernels and length of filters must be equal)

**filters ( Arbitary )** => Must be a list/tuple/array, if blockType is 'type1' , length of filters must be 2 , if blockType is 'type2' , length of filters must be 3. 

**NOTE:** If filKerSelection is 'auto' you can't change the filters you must change filKerSelection to 'manual'

**kernels ( Arbitary )** => Must be a list/tuple/array, if blockType is 'type1' , length of kernels must be 2 , if blockType is 'type2' , length of kernels must be 3. 

**NOTE:** If filKerSelection is 'auto' you can't change the filters you must change filKerSelection to 'manual'

**filKerSelection ( Arbitary )** => default is 'auto' , options: auto/manual

**useStrideFirst ( Arbitary )** => default is False , if True , strides will apply to first convolutional layer

**strides ( Arbitary )** => default is (2,2) , if useStrideFirst is False , it's unnecessary 
