from tensorflow.keras.layers import Layer , Reshape
from tensorflow.math import multiply

class elementWiseMul(Layer):

    def __init__(self , filters, **kwargs):

        super(elementWiseMul , self).__init__(**kwargs)
        self.filters = filters

    def build(self , shape):

        self.shape = shape

        self.kernel = self.add_weight("kernel" , shape = (1 , self.filters))

    def call(self , input):

        shape = list(self.shape[1:])

        shape.append(1)

        x = Reshape(shape)(input)

        return multiply(x , self.kernel)



