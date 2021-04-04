from tensorflow.keras.layers import DepthwiseConv2D , Conv2D , BatchNormalization , Activation , Add 
import tensorTools.__callCounter as callCounter
from reluN import ReluN

def mobileNetV2Block(x , outputChannel , depthWiseKernel = (3,3) , t = 6 , useStride = True , strides = (2,2)):

    expand = x.shape[-1] * t

    y = Conv2D(expand , (1,1) , use_bias = False ,name = '1_1Conv%d' % callCounter.counter)(x)

    y = BatchNormalization(name = 'BN%d' % callCounter.counter)(y)

    y = ReluN(6 , name = '6Relu%d' % callCounter.counter)(y)

    callCounter.counter += 1

    if(useStride):

        y = DepthwiseConv2D(depthWiseKernel , use_bias = False , padding = 'same' , strides = strides , name = "%d_%dDwConv%d/%d" % (depthWiseKernel[0] , depthWiseKernel[1] , callCounter.counter , strides[0]))(y)

    else:

        y = DepthwiseConv2D(depthWiseKernel , use_bias = False , padding = 'same' ,name = "%d_%dDwConv%d" % (depthWiseKernel[0] , depthWiseKernel[1] , callCounter.counter))(y)

    y = BatchNormalization(name = 'BN%d' % callCounter.counter)(y)

    y = ReluN(6 , name = '6Relu%d' % callCounter.counter)(y)

    callCounter.counter += 1

    y = Conv2D(outputChannel , (1,1) , use_bias = False , name = '1_1Conv%d' % callCounter.counter)(y)

    if(str(y.shape) == str(x.shape)):

        y = Add(name = 'Add%d' % callCounter.counter)([x ,y])

        callCounter.counter += 1

        return y

    else:

        callCounter.counter += 1

        return y

    
    