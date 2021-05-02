import math
import numpy as np

def sum_squares(ydata, yhat):
    ''' Return complete sum of squared error over a dataset. PRML Eqn 5.11'''
    # PRML eqn 5.11
    ans = 0.0
    for n in range(ydata.shape[1]):
        ans += np.linalg.norm(ydata[:,n] - yhat[:,n])**2
    return ans/2

def mse(ydata, yhat):
    ''' Return mean squared error over a dataset. '''
    return sum_squares(ydata,yhat)/ydata.shape[1]

def cross_entropy(ydata, yhat):
    ''' Return cross entropy of a dataset.  See PRML eqn 5.24'''
    ans = 0
    for n in range(ydata.shape[1]):
        for k in range(ydata.shape[0]):
            if (ydata[k][n] * yhat[k][n] > 0):
                ans -= ydata[k][n] * math.log(yhat[k][n])
    return ans;

def mce(ydata, yhat):
    ''' Return mean cross entropy over a dataset. '''
    return cross_entropy(ydata, yhat)/ydata.shape[1]

def fscore(ydata, yhat):
    true_positive = 0#supposed to be true and predicted true
    false_positive = 0#supposed to be false but predicted true
    false_negative = 0#supposed to be true but predicted false

    for n in range(ydata.shape[1]):
        if np.argmax(ydata[:,n]) == 0:
            # supposed to be false
            if np.argmax(yhat[:,n]) ==0:
                # predicted false
                continue
            else:
                # predicted true
                false_positive+=1
        else :
            # supposed to be true
            if np.argmax(yhat[:,n]) ==0:
                # predicted false
                false_negative+=1
            else:
                # predicted true
                true_positive+=1

    if true_positive == 0:
        return 0
    P = true_positive / (true_positive + false_positive)
    R = true_positive / (true_positive + false_negative)
    return 2*P*R/(P+R)
def accuracy(ydata, yhat):
    ''' Return accuracy over a dataset. '''
    correct = 0
    for n in range(ydata.shape[1]):
        if (np.argmax(ydata[:,n]) == np.argmax(yhat[:,n])):
            correct += 1
    return correct / ydata.shape[1]
