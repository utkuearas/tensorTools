from tensorflow.math import minimum , maximum
from tensorflow.keras.layers import Layer


class ReluN(Layer):

    def __init__(self, n , **kwargs):

        super(ReluN , self).__init__(**kwargs)
        self.n = n


    def build(self, shape):

        pass

    def call(self,x):

        return minimum(maximum(x,0),self.n)