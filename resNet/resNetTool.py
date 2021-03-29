from tensorflow.keras.layers import Conv2D , Activation , BatchNormalization , Add
import tensorTools.__callCounter as callCounter

def residualBlock(x , blockType = 'type2', filters = None , filKerSelection = 'auto' , kernels = None , useStrideFirst = False , strides = (2,2)):

    if(blockType == 'type1'):

        if(filters is not None or kernels is not None):

            filKerSelection = 'manual'

        if(filKerSelection == 'auto'):

            channels = x.shape[-1]

            if(channels <= 4):

                filters = (64,64)

            else:

                filters = (channels, channels)

            kernels = ((3,3),(3,3))

        elif(filKerSelection == 'manual'):

            if(kernels is None or filters is None):

                raise Exception('You must give both the kernels and the filters parameters when you choose manual selection')

        else:

            raise Exception('Invalid filterSelection Mode\nOptions: "auto" (default) , "manual"')

        if(len(kernels) != 2 or len(filters) != 2):

            raise Exception('The length of kernels and filters must equal 2\nGiven kernels: %s\nGiven filters: %s' % (str(kernels) , str(filters)))

        y = BatchNormalization(name = 'BN%d' % callCounter.counter)(x)

        y = Activation('relu',name = 'Relu%d' % callCounter.counter)(y)

        if(useStrideFirst):

            y = Conv2D(filters[0] , kernels[0] ,use_bias = False, padding = 'same' , strides = strides , name = '%d_%dConv%d/%d' % (kernels[0][0] , kernels[0][1] , callCounter.counter , strides[0]))(y)

        else:

            y = Conv2D(filters[0] , kernels[0] , use_bias = False,  padding = 'same' , name = '%d_%dConv%d' % (kernels[0][0] , kernels[0][1] , callCounter.counter))(y)

        callCounter.counter += 1

        y = BatchNormalization(name = 'BN%d' % callCounter.counter)(y)

        y = Activation('relu', name = 'Relu%d' % callCounter.counter)(y)

        y = Conv2D(filters[1] ,kernels[0] ,use_bias = False , padding = 'same' , name = '%d_%dConv%d' % (kernels[1][0] , kernels[1][1] , callCounter.counter))(y)

        if(str(x.shape) != str(y.shape)):

            callCounter.counter += 1

            return y

        else:

            callCounter.counter += 1

            if(channels <= 3):

                return y

            y = Add(name = 'Add%d' % callCounter.counter)([x , y])

            return y
    
    elif(blockType == 'type2'):

        if(filKerSelection == 'auto'):

            channels = x.shape[-1]

            if(channels <= 4):

                filters = (16,16,64)

            else:

                filters = (channels / 4 , channels / 4 , channels)

            kernels = ((1,1) , (3,3) , (1,1))

        elif(filKerSelection == 'manual'):

            if(kernels is None or filters is None):

                raise Exception('You must give both the kernels and the filters parameters when you choose manual selection')

        else:

            raise Exception('Invalid filterSelection Mode\nOptions: "auto" (default) , "manual"')

        if(len(kernels) != 3 or len(filters) != 3):

            raise Exception('The length of kernels and filters must equal 2\nGiven kernels: %s\nGiven filters: %s' % (str(kernels) , str(filters)))

        y = BatchNormalization(name = 'BN%d' % callCounter.counter)(x)

        y = Activation('relu',name = 'Relu%d' % callCounter.counter)(y)

        if(useStrideFirst):

            y = Conv2D(filters[0] , kernels[0], use_bias = False , padding = 'same' , strides = strides , name = '%d_%dConv%d/%d' % (kernels[0][0] , kernels[0][1] , callCounter.counter , strides[0]))(y)

        else:

            y = Conv2D(filters[0] , kernels[0] , use_bias = False , padding = 'same' , name = '%d_%dConv%d' % (kernels[0][0] , kernels[0][1] , callCounter.counter))(y)

        callCounter.counter += 1

        y = BatchNormalization(name = 'BN%d' % callCounter.counter)(y)

        y = Activation('relu', name = 'Relu%d' % callCounter.counter)(y)

        y = Conv2D(filters[1] , kernels[1] , use_bias = False , padding = 'same' , name = '%d_%dConv%d' % (kernels[1][0] , kernels[1][1] , callCounter.counter))(y)

        callCounter.counter += 1

        y = BatchNormalization(name = 'BN%d' % callCounter.counter)(y)

        y = Activation('relu', name = 'Relu%d' % callCounter.counter)(y)

        y = Conv2D(filters[2] , kernels[2] , use_bias = False , name = '%d_%dConv%d' % (kernels[2][0] , kernels[2][1] , callCounter.counter))(y)

        if(str(x.shape) != str(y.shape)):

            callCounter.counter += 1

            return y

        else:

            if(channels <= 3):

                callCounter.counter += 1

                return y

            y = Add(name = 'Add%d' % callCounter.counter)([x , y])

            callCounter.counter += 1

            return y

    elif(blockType == 'custom'):

        if(kernels is None or filters is None):

            raise Exception('You must give both the kernels and the filters parameters when you choose custom blockType')

        if(len(filters) != len(kernels) and len(kernels) != 1):

            raise Exception('Wrong Kernels or Filters\nLength of filters and length of kernels must be same\nGiven Kernels: %s\nGiven Filters: %s' % (str(kernels) , str(filters)))

        if(len(kernels) == 1):

            kernels = tuple([kernels[0] for i in filters])

        first = True

        for count in range(len(filters)):

            if(first):

                y = BatchNormalization(name = 'BN%d' % callCounter.counter)(x)

                first = False

            else:

                y = BatchNormalization(name = 'BN%d' % callCounter.counter)(y)

            y = Activation('relu' , name = 'Relu%d' % callCounter.counter)(y)

            if(useStrideFirst):
                
                y = Conv2D(filters[count] , kernels[count] , use_bias=False ,name = '%d_%dConv%d/%d' % (kernels[count][0] , kernels[count][1] , callCounter.counter , strides[0]) ,padding = 'same' , strides = strides)(y)

                useStrideFirst = False

            else:

                y = Conv2D(filters[count] , kernels[count], use_bias = False, name = '%d_%dConv%d' % (kernels[count][0] , kernels[count][1] , callCounter.counter) , padding = 'same')(y)

            callCounter.counter += 1

        if(str(x.shape) != str(y.shape)):

            return y

        else:

            y = Add(name = 'Add%d' % callCounter.counter)([x , y])

            return y
        

    else:

        raise Exception('Invalid type\nType we got: %s' % blockType)





