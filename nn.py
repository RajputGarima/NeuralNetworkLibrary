import pandas as pd
import numpy as np
import timeit
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import sys


config_file = sys.argv[1]
xtrain_file = sys.argv[2]
ytrain_file = sys.argv[3]
xtest_file = sys.argv[4]
ytest_file = sys.argv[5]

# -- load data
df = pd.read_csv(xtrain_file, header=None, dtype=int)
df1 = pd.read_csv(ytrain_file, header=None, dtype=int)

df2 = pd.read_csv(xtest_file, header=None, dtype=int)
df3 = pd.read_csv(ytest_file, header=None, dtype=int)

X_train = np.asarray(df, dtype=int)
y_train = np.asarray(df1, dtype=int)

X_test = np.asarray(df2, dtype=int)
y_test = np.asarray(df3, dtype=int)

# -- neural network

f = open(config_file, mode='r')
file_ctr = 0
layer_list = []
batch_size = 0
output_layer = 0
intermediate = []
nl = 0
tol = 10 ** -4
adaptive = False
relu_active = False

for line in f:
    line = line.rstrip()
    file_ctr += 1
    if file_ctr == 1:
        layer_list.append(int(line))
    elif file_ctr == 2:
        output_layer = int(line)
    elif file_ctr == 3:
        batch_size = int(line)
    elif file_ctr == 4:
        nl = int(line)
    elif file_ctr == 5:
        arr = line.split(' ')
        for i in range(nl):
            intermediate.append(int(arr[i]))
    elif file_ctr == 7:
        if line == 'variable':
            adaptive = True
    elif file_ctr == 6:
        if line == 'relu':
            relu_active = True
f.close()

for i in intermediate:
    layer_list.append(i)
layer_list.append(output_layer)


start = timeit.default_timer()
weights = []
bias = []
y_predicted = np.zeros((len(y_test), 1))
y_actual = np.zeros((len(y_test), 1))

print(layer_list)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(- x))


def relu(z):
    return z * (z > 0)


def derivative_sigmoid(x):
    return x * (1 - x)


def derivative_relu(z):
    return 1 * (z > 0)


def loss_function(a, y):
    return 0.5 * np.linalg.norm(a - y) ** 2


def derivative_loss_function(a, y):
    return a - y


def init_w_b():
    for i, j in zip(layer_list[:-1], layer_list[1:]):
        weights.append(np.random.randn(j, i) / np.sqrt(i))
        bias.append(np.random.randn(j, 1))


def feed_forward(x):
    a = x
    output = [a]
    for i in range(len(layer_list) - 1):
        z = np.dot(weights[i], a) + bias[i]
        if relu_active and i < len(layer_list) - 2:
            a = relu(z)
        else:
            a = sigmoid(z)
        output.append(a)
    return output


def back_propagation(a, output):
    del_w = [0] * (len(layer_list) - 1)
    del_b = [0] * (len(layer_list) - 1)
    for i in range(1, len(layer_list)):
        if relu_active and -i != -1:
            del_out = derivative_relu(output[-i])
        else:
            del_out = derivative_sigmoid(output[- i])
        if -i == -1:
            a = np.reshape(a, (a.shape[0], 1))
            temp = derivative_loss_function(output[-1], a)
            delta = temp * del_out
        else:
            delta = del_out * np.dot(weights[-i + 1].T, delta)
        # Gradients at this layer
        del_w[-i] = np.dot(delta, np.transpose(output[-i - 1]))
        del_b[-i] = delta
    return del_w, del_b


def total_error(X_train, y_train):
    return sum(loss_function(feed_forward(x)[-1], y) for x, y in zip(X_train, y_train)) / len(X_train)


def train_nn(X_train, y_train, batch_size, adaptive=False):
    init_w_b()
    eeta = 0.1
    epochs = 1000
    error_threshold = 10 ** -12
    X_train = np.array([x.reshape(-1, 1) for x in X_train])
    indices = np.arange(len(X_train))
    epoch = 1
    error = np.inf
    error_old = np.inf
    while True:
        print(epoch)
        epoch += 1
        # shuffle the indices of the data at each epoch
        np.random.shuffle(indices)

        # Iterate over batches of data
        for i in range(0, len(X_train), batch_size):

            batch = indices[i:i + batch_size]
            Xb, Yb = X_train[batch], y_train[batch]
            dw, db = [0] * len(layer_list), [0] * len(layer_list)

            # Compute gradient
            for xb, yb in zip(Xb, Yb):

                layer_outputs = feed_forward(xb)
                gradients = back_propagation(yb, layer_outputs)

                # adding gradient of a particular batch
                for i, (dw_i, db_i) in enumerate(zip(*gradients)):

                    if dw[i] is 0:  # Initialization condition
                        dw[i] = np.zeros(weights[i].shape)
                        db[i] = np.zeros(bias[i].shape)

                    dw[i] += dw_i / len(Xb)
                    db[i] += db_i / len(Xb)

            # the gradient descent
            for l in range(len(layer_list) - 1):
                weights[l] -= eeta * dw[l]
                bias[l] -= eeta * db[l]

        # Compute error on training data
        error_third_last = error_old
        error_old = error
        error = total_error(X_train, y_train)

        if adaptive:
            if abs(error_old - error) <= tol and abs(error_third_last - error_old) <= tol:
                eeta = eeta/5

        if abs(error_old - error) <= error_threshold:
            print("Error threshold reached\n\n")
            stop = timeit.default_timer()
            print("Training time: ", stop - start)
            break

        elif epoch == epochs:
            print("Maximum epochs reached\n\n")
            stop = timeit.default_timer()
            print("Training time:", stop - start)
            break


train_nn(X_train, y_train, batch_size, adaptive)


def predict(x):
        o = feed_forward(x.reshape(-1, 1))
        p = np.asarray(o[-1])
        return np.argmax(p)


def datasets_prediction():
    c = 0
    for i in range(len(y_train)):
        target_label = np.where(y_train[i] == 1)[0]
        if target_label[0] == predict(X_train[i]):
            c += 1
    print("Train accuracy: ", c/len(y_train))
    c = 0
    for i in range(len(y_test)):
        target_label = np.where(y_test[i] == 1)[0]
        y_actual[i] = target_label[0]
        temp = predict(X_test[i])
        y_predicted[i] = temp
        if target_label[0] == temp:
            c += 1
    print("Test accuracy: ", c / len(y_test))


def plot_confusion_matrix(y_actual, y_predicted,  classes, title,  cmap=plt.cm.Blues):
    cm = confusion_matrix(y_actual, y_predicted)
    # classes = classes[unique_labels(y_actual, y_predicted)]
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


datasets_prediction()
class_names = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
plot_confusion_matrix(y_actual, y_predicted, classes=class_names, title='Confusion matrix: Testing data')
plt.show()

# res_train = [0.50271819, 0.5106757, 0.5262295, 0.53882447, 0.54375849]
# res_test = [0.496054, 0.500934, 0.510363, 0.527561, 0.537074]
# neurons = [5, 10, 15, 20, 25]
#
# line1, = plt.plot(neurons, res_train, "b", label="Training accuracy")
# line2, = plt.plot(neurons, res_test, "r", label="Testing accuracy ")
# plt.legend(handles=[line1, line2])
# plt.ylabel('Accuracy')
# plt.xlabel("Number of hidden layer units")
# plt.title("Number of neurons v/s accuracy [Batch size: 100]")
# plt.show()
