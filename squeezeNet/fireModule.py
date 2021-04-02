from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D , Activation , Concatenate , Add
import tensorTools.__callCounter as callCounter


def fireModule(x,squeeze = 16 , expand = 64 , kernels = ((1,1) , (3,3)) , bypass = False , complexBypass = False):

    if(len(kernels) != 2):

        raise Exception('The length of kernels must equal 2\nGiven kernel: %s' % str(kernels))

    lowKernel = kernels[0]
    highKernel = kernels[1]

    y = Conv2D(squeeze , lowKernel , padding = 'same' , name = '%d_%dConv%dSqueeze' % (lowKernel[0],lowKernel[1], callCounter.counter))(x)

    y = Activation('relu' , name = 'Relu1%d' %callCounter.counter)(y)

    left = Conv2D(expand , highKernel , padding = 'same' , name = '%d_%dConv%dLeft' % (highKernel[0],highKernel[1] , callCounter.counter))(y)

    left = Activation('relu', name = 'Relu2%d' %callCounter.counter)(left)

    right = Conv2D(expand , lowKernel , padding = 'same' , name = '%d_%dConv%dRight' % (lowKernel[0],lowKernel[1], callCounter.counter))(y)

    right = Activation('relu', name = 'Relu3%d' %callCounter.counter)(right)

    concat = Concatenate(axis = -1)([left , right])

    if(complexBypass):

        additionalConv = Conv2D(expand * 2 , (1,1) , name = '1_1AdditionalConv%d' %  callCounter.counter)(x)

        concat = Add()([concat , additionalConv])

    elif(bypass):

        if(str(x.shape) != str(concat.shape)):

            raise Exception('The shape of input and output must be same.\nInput shape: %s\nOutput shape: %s' % (str(x.shape) , str(concat.shape))) 

        concat = Add(name = 'Add%d'%callCounter.counter)([concat , x])

    callCounter.counter += 1

    return concat

