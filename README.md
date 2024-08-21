# mnist_sparse
ML classifier on MNIST dataset with sparsely connected layers. Build with Python and Pytorch

# Overview and specs
The default model runs for 20 epochs in training and has app. 94% accuracy after that time. It has two hidden layers. The input layer is 28x28 picture expressed in grey scale values from mnist_train.csv or mnist_test.csv. The hidden layers have 512 and 10 nodes respectively. The output layer has 10 nodes corresponding to 10 classification sections for number 0 through 9. The activation function between layers is always ReLU. Cross entropy loss function is at use here. The optimizer is SGD the whole thing, while learning rate is set as default to 0.1. All layers of this model are partially connected (aka. sparse) with 50% (0.5) of connections disconnected.
