import tensorTools.__callCounter as callCounter
from tensorflow.keras.layers import DepthwiseConv2D , Conv2D , BatchNormalization , Activation 


def mobileNetV1Block(x , depthWiseKernel = (3,3) , filters = None , useStride = False , strides = (2,2)):

    if(filters == None):

        filters = x.shape[-1] * 2

    if(useStride):

        y = DepthwiseConv2D(depthWiseKernel , padding = 'same' , strides = strides , use_bias = False , name = '%d_%dDwConv%d' % (depthWiseKernel[0] , depthWiseKernel[1] , callCounter.counter))(x)

    else:

        y = DepthwiseConv2D(depthWiseKernel , padding = 'same' , strides = strides , use_bias = False , name = '%d_%dDwConv%d' % (depthWiseKernel[0] , depthWiseKernel[1] , callCounter.counter))(x)

    y = BatchNormalization(name = 'BN%d' % callCounter.counter)(y)

    y = Activation('relu' , name = 'Relu%d' % callCounter.counter)(y)

    callCounter.counter += 1

    y = Conv2D(filters , (1,1) , use_bias = False , name = '1_1Conv%d' %callCounter.counter)(y)

    return y