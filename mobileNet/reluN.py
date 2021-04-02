from tensorflow.math import minimum , maximum
from tensorflow.keras.layers import Layer


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