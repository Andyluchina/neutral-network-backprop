import numpy as np
import pickle
import math

''' An implementation of an MLP with a single layer of hidden units. '''
class MLP:
    __slots__ = ('W1', 'b1', 'a1', 'z1', 'W2', 'b2', 'din', 'dout', 'hidden_units')

    def __init__(self, din, dout, hidden_units):
        ''' Initialize a new MLP with tanh activation and softmax outputs.

        Params:
        din -- the dimension of the input data
        dout -- the dimension of the desired output
        hidden_units -- the number of hidden units to use

        Weights are initialized to uniform random numbers in range [-1,1).
        Biases are initalized to zero.

        Note: a1 and z1 can be used for caching during backprop/evaluation.

        '''
        self.din = din
        self.dout = dout
        self.hidden_units = hidden_units

        self.b1 = np.zeros((self.hidden_units, 1))
        self.b2 = np.zeros((self.dout, 1))
        self.W1 = 2*(np.random.random((self.hidden_units, self.din)) - 0.5)
        self.W2 = 2*(np.random.random((self.dout, self.hidden_units)) - 0.5)


    def save(self,filename):
        with open(filename, 'wb') as fh:
            pickle.dump(self, fh)

    def tanh(self, z):
        try:
            ans = math.exp(z)
        except OverflowError:
            ans = float('inf')
        try:
            ansn = math.exp(-z)
        except OverflowError:
            ansn = float('inf')
        numerator = ans - ansn
        denominator = ans + ansn

        if denominator == 0:
            return float('inf')
        elif denominator == float('inf'):
            return 0
        return (numerator/denominator)

    def derivative_of_tanh(self, z):
        return (1 - math.pow(self.tanh(z), 2))

    def load_mlp(filename):
        with open(filename, 'rb') as fh:
            return pickle.load(fh)

    def eval(self, xdata):
        ''' Evaluate the network on a set of N observations.

        xdata is a design matrix with dimensions (Din x N).
        This should return a matrix of outputs with dimension (Dout x N).
        See train_mlp.py for example usage.
        '''
        # print(self.b1) #hidden units
        # print(self.b2) #output
        # print(self.W1)
        # print(self.W2)
        # print(self.W1.dot(xdata))

        hidden_unit_values = self.W1.dot(xdata)
        # bias calculation layer 1
        for y in range(hidden_unit_values.shape[1]):
            for x in range(hidden_unit_values.shape[0]):
                hidden_unit_values[x][y] += self.b1[x][0]
        self.a1 = hidden_unit_values
        self.z1 = np.copy(hidden_unit_values)
        # activation
        for x in range(hidden_unit_values.shape[0]):
            for y in range(hidden_unit_values.shape[1]):
                self.z1[x][y] = self.tanh((self.z1[x][y]))
        output = self.W2.dot(self.z1)

        # bias calculation layer 2
        for y in range(output.shape[1]):
            for x in range(output.shape[0]):
                output[x][y] += self.b2[x][0]

        # print(output)
        # apply softmax to output
        for y in range(output.shape[1]):
            sum_of_column = 0
            for x in range(output.shape[0]):
                sum_of_column += math.exp(output[x][y])
            for x in range(output.shape[0]):
                output[x][y] = math.exp(output[x][y])/sum_of_column
        return output

    def sgd_step(self, xdata, ydata, learn_rate):
        ''' Do one step of SGD on xdata/ydata with given learning rate. '''
        #update self.W1 and self.W2 and self.b1 and self.b2
        gradients = self.grad(xdata, ydata)
        self.W1 -= learn_rate*gradients[0]
        self.b1 -= learn_rate*gradients[1]
        self.W2 -= learn_rate*gradients[2]
        self.b2 -= learn_rate*gradients[3]

    def grad(self, xdata, ydata):
        ''' Return a tuple of the gradients of error wrt each parameter.
        Result should be tuple of four matrices:
          (dE/dW1, dE/db1, dE/dW2, dE/db2)
          # print(self.b1) bias weights layer 1
          # print(self.b2) bias weights layer 2

        Note:  You should calculate this with backprop,
        but you might want to use finite differences for debugging.
        '''
        yhat = self.eval(xdata)

        delta_nplus1 = yhat - ydata
        # dw2
        dw2 = delta_nplus1.dot(self.z1.transpose())*(1/self.z1.shape[1])
        # db2
        h_aj1 = self.tanh(1)
        bias_vector2 = np.full((self.z1.shape[1],1),h_aj1)
        db2 = delta_nplus1.dot(bias_vector2)*(1/self.z1.shape[1])
        # dw1
        h_aj0 = np.copy(xdata)
        for x in range(h_aj0.shape[0]):
            for y in range(h_aj0.shape[1]):
                h_aj0[x][y] = self.tanh((h_aj0[x][y]))

        h_prime_ak1 = np.copy(self.a1)
        for x in range(h_prime_ak1.shape[0]):
            for y in range(h_prime_ak1.shape[1]):
                h_prime_ak1[x][y] = self.derivative_of_tanh((h_prime_ak1[x][y]))

        # delta_nplus1 and self.W2
        sum_delta2_w2 = self.W2.transpose().dot(delta_nplus1)
        delta1 = np.multiply(h_prime_ak1, sum_delta2_w2)
        dw1 = delta1.dot(h_aj0.transpose())*(1/self.z1.shape[1])
        # db1 h_aj0 the same
        delta1_first_term = self.derivative_of_tanh(1)
        bias_vector1 = np.full((self.z1.shape[1],1), delta1_first_term)
        db1 = delta1.dot(bias_vector1)*(1/self.z1.shape[1])
        return (dw1, db1, dw2, db2)
