# NeuralNetworkLibrary

Built my own **generic neural network library** that takes 5 parameters as follows:
1. config.txt
2. training_data_features(one hot encoded)
3. training_data_targetValue(one hot encoded)
4. testing_data_features(one hot encoded)
5. testing_data_targetValue(one hot encoded)

It outputs the accuracy on test data and also plots the confusion matrix to give a visual representation of the results of classifier. <br/>

A sample **config.txt** is attached. It has 7 parameters:
1. Number of input features
2. Number of output classes
3. Batch size for stochastic gradient descent
4. No of hidden layers, say n
5. n space seperated integers denoting number of neurons in each hidden layer
6. Activation function: sigmoid/relu
7. Learning rate: fixed/variable
