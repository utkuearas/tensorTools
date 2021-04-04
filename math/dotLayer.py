from tensorflow.keras.layers import Layer
from tensorflow.linalg import matmul

class trainableDot(Layer):

    def __init__(self, filters , **kwargs):

        super(trainableDot , self).__init__(kwargs)
        self.filters = filters

    def build(self , shape):

        self.kernel = self.add_weight("kernel" , shape = (shape[-1] , self.filters))

    def call(self, input):

        return matmul(input , self.kernel)

