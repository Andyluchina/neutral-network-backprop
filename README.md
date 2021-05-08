# neutral-network-backprop

analysis.py is the original file with the implementation of f score added.
dataproc.py is unchanged.
mlp.py is the main implementation of the neural network with all three requirement functions
modelFile is the save model file.
train_mlp.py is modified with f score and plotting.


Usage
The usage of the code is unchanged, and a example command is provided below:
python .\train_mlp.py --train_file ../data/htru2.train --dev_file ../data/htru2.dev --hidden_units 11 --epochs 20 --batch 5


Recommendations of the model parameters
Thoughts on Epoch: in general, definitely the more epochs, the better in terms of getting the best results, but more epochs on more hidden units could be more time consuming and unnecessary. In this case, 100 epoch does give a better result than 50, but the improvement is small. I would recommend 100 for the best result.
Thoughts on learn_rate: seems like 0.1 is the best one as well. A smaller learn rate could not achieve improvements. A bigger learn rate actually makes the results worse after more epoch.
Thoughts on batch: batch seems to be mixed bag. A high or low batch would not yield a higher accuracy, a batch of 5 is appropriate for this particular problem.
Thoughts on hidden_units: after experimenting on a couple of hidden units, 10 seems to be the best number. The reason why I don’t believe I am overfitting is that the dev data follows the similar pattern as the f score does not get worse.
