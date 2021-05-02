import argparse
import numpy as np
import random
import analysis
import dataproc
import mlp
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Program to build and train a neural network.')
    parser.add_argument('--train_file', default=None, help='Path to the training data.')
    parser.add_argument('--dev_file', default=None, help='Path to the development data.')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train. (default 10)')
    parser.add_argument('--learn_rate', type=float, default=1e-1, help='The learning rate to use for SGD (default 1e-1).')
    parser.add_argument('--hidden_units', type=int, default=0, help='The number of hidden units to use. (default 0)')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for SGD. (default 1)')
    args = parser.parse_args()
    epoch_axis_train = []
    fscore_axis_train = []
    epoch_axis_dev = []
    fscore_axis_dev = []

    # Load training and development data and convert labels to 1-hot representation.
    xtrain, ytrain = dataproc.load_data(args.train_file)
    ytrain = dataproc.to_one_hot(ytrain, int(1+np.max(ytrain[0,:])))
    if (args.dev_file is not None):
        xdev, ydev = dataproc.load_data(args.dev_file)
        ydev = dataproc.to_one_hot(ydev,int(1+np.max(ytrain[0,:])))

    # Record dimensions and size of dataset.
    N = xtrain.shape[1] # size of the data
    din = xtrain.shape[0]
    dout = ytrain.shape[0]
    # print(xtrain)
    # print(ytrain)
    batch_size = args.batch_size
    if (batch_size == 0):
        batch_size = N

    # Create an MLP object for training.
    nn = mlp.MLP(din, dout, args.hidden_units)
    yhat = nn.eval(xtrain)

    try:
        nnBest = mlp.MLP.load_mlp("modelFile")
        ybest = nnBest.eval(xdev)
    except:
        ybest = nn.eval(xdev)

    # Evaluate MLP after initialization; yhat is matrix of dim (Dout x N).


    best_train = (analysis.mse(ytrain, yhat),
                  analysis.mce(ytrain, yhat),
                  analysis.accuracy(ytrain, yhat)*100,
                  analysis.fscore(ytrain, yhat))
    print('Initial conditions~~~~~~~~~~~~~')
    print('mse(train):  %f'%(best_train[0]))
    print('mce(train):  %f'%(best_train[1]))
    print('acc(train):  %f'%(best_train[2]))
    print('fscore:  %f'%(best_train[3]))
    print('')
    epoch_axis_train.append(0)
    fscore_axis_train.append(best_train[3])
    epoch_axis_dev.append(0)
    fscore_axis_dev.append(best_train[3])

    if (args.dev_file is not None):
        best_dev = (analysis.mse(ydev, ybest),
                      analysis.mce(ydev, ybest),
                      analysis.accuracy(ydev, ybest)*100,
                      analysis.fscore(ydev, ybest))
        print('mse(dev):  %f'%(best_dev[0]))
        print('mce(dev):  %f'%(best_dev[1]))
        print('acc(dev):  %f'%(best_dev[2]))
        print('fscore:  %f'%(best_dev[3]))

    for epoch in range(args.epochs):
        for batch in range(int(N/batch_size)):
            ids = random.choices(list(range(N)), k=batch_size)
            xbatch = np.array([xtrain[:,n] for n in ids]).transpose()
            ybatch = np.array([ytrain[:,n] for n in ids]).transpose()
            nn.sgd_step(xbatch, ybatch, args.learn_rate)

        yhat = nn.eval(xtrain)
        train_ss = analysis.mse(ytrain, yhat)
        train_ce = analysis.mce(ytrain, yhat)
        train_acc = analysis.accuracy(ytrain, yhat)*100
        train_fscore = analysis.fscore(ytrain, yhat)
        best_train = (min(best_train[0], train_ss), min(best_train[1], train_ce), max(best_train[2], train_acc), max(best_train[3], train_fscore))

        print('After %d epochs ~~~~~~~~~~~~~'%(epoch+1))
        print('mse(train):  %f  (best= %f)'%(train_ss, best_train[0]))
        print('mce(train):  %f  (best= %f)'%(train_ce, best_train[1]))
        print('acc(train):  %f  (best= %f)'%(train_acc, best_train[2]))
        print('fscore(train):  %f  (best= %f)'%(train_fscore, best_train[3]))
        # epoch_axis_train = []
        # fscore_axis_train = []
        # epoch_axis_dev = []
        # fscore_axis_dev = []
        epoch_axis_train.append(epoch+1)
        fscore_axis_train.append(train_fscore)

        if (args.dev_file is not None):
            yhat = nn.eval(xdev)
            dev_ss = analysis.mse(ydev, yhat)
            dev_ce = analysis.mce(ydev, yhat)
            dev_acc = analysis.accuracy(ydev, yhat)*100
            dev_fscore = analysis.fscore(ydev, yhat)
            if dev_fscore >= best_dev[3]:
                print("best saved")
                nn.save('modelFile')
            best_dev = (min(best_dev[0], dev_ss), min(best_dev[1], dev_ce), max(best_dev[2], dev_acc), max(best_dev[3], dev_fscore))
            print('mse(dev):  %f  (best= %f)'%(dev_ss, best_dev[0]))
            print('mce(dev):  %f  (best= %f)'%(dev_ce, best_dev[1]))
            print('acc(dev):  %f  (best= %f)'%(dev_acc, best_dev[2]))
            print('fscore(dev):  %f  (best= %f)'%(dev_fscore, best_dev[3]))
            epoch_axis_dev.append(epoch+1)
            fscore_axis_dev.append(dev_fscore)
        print('')

    plt.plot(epoch_axis_train, fscore_axis_train)
    plt.ylabel('F socre')
    plt.xlabel('Epoch')
    plt.title("Train file F score over time")
    plt.figtext(0, 0, "The best accuracy is "+str(best_train[2])+", and the best f score is "+str(best_train[3]), fontsize = 10)
    plt.show()

    plt.plot(epoch_axis_dev, fscore_axis_dev)
    plt.ylabel('F socre')
    plt.xlabel('Epoch')
    plt.title("Dev file F score over time")
    plt.figtext(0, 0, "The best accuracy is "+str(best_dev[2])+", and the best f score is "+str(best_dev[3]), fontsize = 10)
    plt.show()


if __name__ == '__main__':
    main()
