
import tensorflow as tf
import numpy as np

def load_data(num_classes=10):
    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xtrain /= 255 # normalize
    xtest /= 255  # normalize
    ytrain = np.eye(num_classes)[ytrain] # one hot encoding
    ytest = np.eye(num_classes)[ytest]   # one hot encoding
    return xtrain, ytrain, xtest, ytest

def next_batch(batch_size, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def build_model(X, num_input, num_output):
    n0 = num_input   # MNIST data input (img shape: 28*28)
    n1 = 256         # number of neurons in 1st hidden layer
    n2 = num_output  # output layer

    W1 = tf.Variable(tf.random_normal([n0, n1]))
    B1 = tf.Variable(tf.random_normal([n1]))
    W2 = tf.Variable(tf.random_normal([n1, n2]))
    B2 = tf.Variable(tf.random_normal([n2]))

    Y1 = tf.add(tf.matmul(X, W1), B1)
    yhat = tf.add(tf.matmul(Y1, W2), B2)

    return yhat

xtrain, ytrain, xtest, ytest = load_data()

# Parameters
learning_rate = 0.1
num_epoch = 500
batch_size = 128

num_input = xtrain.shape[1]   # image shape: 28*28=784
num_classes = ytrain.shape[1] # MNIST total classes (0-9 digits)

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# construct model
yhat = build_model(X, num_input, num_classes)
pred = tf.nn.softmax(yhat)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=yhat, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# evaluate model
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epoch):
        xbatch, ybatch = next_batch(batch_size, xtrain, ytrain)
        sess.run(train_op, feed_dict={X: xbatch, Y: ybatch})

        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: xbatch, Y: ybatch})
        print("epoch " + str(epoch) + ", loss= " + "{:.4f}".format(loss) + ", acc= " + "{:.3f}".format(acc))

    # Calculate accuracy for MNIST test images
    acc = sess.run(accuracy, feed_dict={X: xtest, Y: ytest})
    print('test acc=' + '{:.3f}'.format(acc))