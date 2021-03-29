from tensorflow.keras.layers import DepthwiseConv2D , Conv2D , BatchNormalization , Activation , Add , Layer
import tensorTools.__callCounter as callCounter
from tensorflow.math import minimum , maximum


class ReluN(Layer):

    def __init__(self, n , name = None):

        super(ReluN , self).__init__()
        self.n = n

        if(name != None):

            self._name = name

    def build(self, shape):

        pass

    def call(self,x):

        return minimum(maximum(x,0),self.n)


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

def mobileNetV2Block(x , outputChannel , depthWiseKernel = (3,3) , filters = None , t = 6 , useStride = True , strides = (2,2)):

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

    
    