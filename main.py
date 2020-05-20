'''
    Topics in Deep Learning Assignment

    @authors
    Hritvik Patel       PES1201700125
    Shreyas BS          PES1201700956
    Archana Prakash     PES1201701543
'''

# ------------------------------------------------ IMPORTS ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf2
import tensorflow.compat.v1 as tf1
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split

tf1.disable_v2_behavior()

# ---------------------------------------- DATASET & PREPROCESSING ----------------------------------------

PATH_OF_FILE = 'labels.csv'
PATH_OF_DATA = 'images/'

labels = []
images = []

file = open(PATH_OF_FILE, 'r')
file = list(file)

for i in range(1, len(file)):
    image = cv2.imread(PATH_OF_DATA + file[i].split(",")[0], cv2.IMREAD_UNCHANGED)
    image = image.flatten()
    image = np.array(image, dtype='float32')
    images.append(image)
    labels.append([int(file[i].split(",")[1])])

images = np.array(images)
labels = np.array(labels, dtype='float32')
train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)

# ---------------------------------------------- AUTOENCODER ----------------------------------------------

num_inputs = 10000    # 100 x 100 pixels
num_hid1 = 5000
num_hid2 = 2500
num_hid3 = num_hid2
num_hid4 = num_hid1
num_output = num_inputs
lr = 0.001
actf = tf2.nn.relu

X = tf1.placeholder(tf2.float32, shape = [None, num_inputs])
initializer = tf1.variance_scaling_initializer()

w1 = tf2.Variable(initializer([num_inputs, num_hid1]), dtype = tf2.float32)
w2 = tf2.Variable(initializer([num_hid1, num_hid2]), dtype = tf2.float32)
w3 = tf2.Variable(initializer([num_hid2, num_hid3]), dtype = tf2.float32)
w4 = tf2.Variable(initializer([num_hid3, num_hid4]), dtype = tf2.float32)
w5 = tf2.Variable(initializer([num_hid4, num_output]), dtype = tf2.float32)

b1 = tf2.Variable(tf2.zeros(num_hid1))
b2 = tf2.Variable(tf2.zeros(num_hid2))
b3 = tf2.Variable(tf2.zeros(num_hid3))
b4 = tf2.Variable(tf2.zeros(num_hid4))
b5 = tf2.Variable(tf2.zeros(num_output))

hid_layer1 = actf(tf2.matmul(X, w1) + b1)
hid_layer2 = actf(tf2.matmul(hid_layer1, w2) + b2)
hid_layer3 = actf(tf2.matmul(hid_layer2, w3) + b3)
hid_layer4 = actf(tf2.matmul(hid_layer3, w4) + b4)
output_layer = actf(tf2.matmul(hid_layer4, w5) + b5)

# loss = tf2.reduce_mean(tf2.square(output_layer - X))
loss = tf2.reduce_mean(tf2.square(tf2.math.log(output_layer + 1) - tf2.math.log(X + 1)))

optimizer = tf1.train.AdamOptimizer(lr)
train = optimizer.minimize(loss)

init = tf1.global_variables_initializer()

num_epoch = 250
batch_size = 105

num_test_images = 10

new_input = []
with tf1.Session() as sess:
    sess.run(init)

    for epoch in range(1, num_epoch + 1):
        num_batches = 8
        i = 0

        for iteration in range(num_batches):
            X_batch = train_data[i * batch_size:(i + 1) * batch_size]
            i += 1
            sess.run(train, feed_dict = {X:X_batch})

        if epoch % 1 == 0:
            train_loss = loss.eval(feed_dict = {X:train_data})
            print("epoch {} loss {}".format(epoch, train_loss))

    results = hid_layer3.eval(feed_dict = {X:images})
    new_input = results
    print(results)

count = 0

x_input = []

for i in range(len(new_input)):
    temp = []

    for j in range(0, 2500, 50):
        temp1 = []

        for k in range(j, j + 50):
            temp1.append(np.array([new_input[i][k]]))
        
        temp.append(np.array(temp1))
    
    x_input.append(np.array(temp))

final = np.array(x_input)
print(final)

# ------------------------------- MULTI-COLUMN CONVOLUTIONAL NEURAL NETWORK -------------------------------

# initializer = tf1.contrib.layers.xavier_initializer()     # Deprecated in tensorflow 2.x
initializer = tf2.keras.initializers.GlorotNormal()

# --- COLUMN 1 ---
# ------ Convolution Layers - weights ------
w11 = tf2.Variable(initializer([9, 9, 1, 16]))
w12 = tf2.Variable(initializer([7, 7, 16, 32]))
w13 = tf2.Variable(initializer([7, 7, 32, 16]))
w14 = tf2.Variable(initializer([7, 7, 16, 8]))

# ------ Convolution Layers - bias ------
b11 = tf2.Variable(tf2.zeros(16))
b12 = tf2.Variable(tf2.zeros(32))
b13 = tf2.Variable(tf2.zeros(16))
b14 = tf2.Variable(tf2.zeros(8))

# --- COLUMN 2 ---
# ------ Convolution Layers - weights ------
w21 = tf2.Variable(initializer([7, 7, 1, 20]))
w22 = tf2.Variable(initializer([5, 5, 20, 40]))
w23 = tf2.Variable(initializer([5, 5, 40, 20]))
w24 = tf2.Variable(initializer([5, 5, 20, 10]))

# ------ Convolution Layers - bias ------
b21 = tf2.Variable(tf2.zeros(20))
b22 = tf2.Variable(tf2.zeros(40))
b23 = tf2.Variable(tf2.zeros(20))
b24 = tf2.Variable(tf2.zeros(10))

# --- COLUMN 3 ---
# ------ Convolution Layers - weights ------
w31 = tf2.Variable(initializer([5, 5, 1, 24]))
w32 = tf2.Variable(initializer([2, 2, 24, 48]))
w33 = tf2.Variable(initializer([2, 2, 48, 24]))
w34 = tf2.Variable(initializer([3, 3, 24, 12]))

# ------ Convolution Layers - bias ------
b31 = tf2.Variable(tf2.zeros(24))
b32 = tf2.Variable(tf2.zeros(48))
b33 = tf2.Variable(tf2.zeros(24))
b34 = tf2.Variable(tf2.zeros(12))
    
# Dense Layers - weights
wd1 = tf2.Variable(initializer([25 * 25 * 30, 1875]))
wd2 = tf2.Variable(initializer([1875, 187]))
wd3 = tf2.Variable(initializer([187, 1]))

# Dense Layers - bias
bd1 = tf2.Variable(tf2.zeros(1875))
bd2 = tf2.Variable(tf2.zeros(187))
bd3 = tf2.Variable(tf2.zeros(1))

# Define 2D convolutional function
def conv2d(x, W, b, strides = 1):
    x = tf2.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
    x = tf2.nn.bias_add(x, b)
    
    return tf2.nn.relu(x)

Xtrain = tf1.placeholder(tf2.float32, shape = (None, 50, 50, 1))
ytrain = tf1.placeholder(tf2.float32, shape = (None, 1))

# --- COLUMN 1 ---
# Convolution layers
conv11 = conv2d(Xtrain, w11, b11) # [50,50,16]
pool11 = tf2.nn.max_pool(conv11, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') # [25,25,16]
conv12 = conv2d(pool11, w12, b12) # [25,25,32]
#pool12 = tf2.nn.max_pool(conv12, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # [25,25,32]
#conv13 = conv2d(pool12, w13, b13) # [25,25,16]
conv13 = conv2d(conv12, w13, b13) # [25,25,16] 
conv14 = conv2d(conv13, w14, b14) # [25,25,8] 


# --- COLUMN 2 ---
# Convolution layers
conv21 = conv2d(Xtrain, w21, b21) # [50,50,20]
pool21 = tf2.nn.max_pool(conv21, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') # [25,25,20]
conv22 = conv2d(pool21, w22, b22) # [25,25,40]
#pool22 = tf2.nn.max_pool(conv22, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # [25,25,40]
#conv23 = conv2d(pool22, w23, b23) # [25,25,20] 
conv23 = conv2d(conv22, w23, b23) # [25,25,20] 
conv24 = conv2d(conv23, w24, b24) # [25,25,10]


# --- COLUMN 3 ---
# Convolution layers
conv31 = conv2d(Xtrain, w31, b31) # [50,50,24]
pool31 = tf2.nn.max_pool(conv31, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') # [25,25,24]
conv32 = conv2d(pool31, w32, b32) # [25,25,48]
#pool32 = tf2.nn.max_pool(conv32, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # [25,25,48]
#conv33 = conv2d(pool32, w33, b33) # [25,25,24] 
conv33 = conv2d(conv32, w33, b33) # [25,25,24] 
conv34 = conv2d(conv33, w34, b34) # [25,25,12]

merged = tf2.concat([conv14, conv24, conv34], -1)
flat = tf2.reshape(merged, [-1, 25 * 25 * 30]) 

fc1 = tf2.add(tf2.matmul(flat, wd1), bd1) # [1875]
fc2 = tf2.add(tf2.matmul(fc1, wd2), bd2) # [187]
out = tf2.add(tf2.matmul(fc2, wd3), bd3) # [1]
out_final = tf2.nn.sigmoid(out)

train_data, test_data, train_labels, test_labels = train_test_split(final, labels, test_size = 0.2)

loss = tf2.keras.losses.binary_crossentropy(ytrain,out_final)

optimizer = tf1.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(loss)

nepochs = 20
batch_size = 105

with tf1.Session() as sess:
    sess.run(tf1.global_variables_initializer())
    sess.run(tf1.local_variables_initializer())

    for epoch in range(nepochs):
        i = 0

        for step in tqdm(range(int(len(train_data) / batch_size))):
            # Batched data
            X_batch = train_data[i * batch_size:(i + 1) * batch_size]
            y_batch = train_labels[i * batch_size:(i + 1) * batch_size]
            i += 1

            # Train model
            feed_dict = {Xtrain: X_batch, ytrain: y_batch}
            sess.run(train_op, feed_dict = feed_dict)

        train_loss = loss.eval(feed_dict = {Xtrain:train_data, ytrain:train_labels})

    results = out_final.eval(feed_dict = {Xtrain:test_data})
    results = results.flatten()
    results = list(map(lambda x: 1 if(x > 0.5) else 0 , results))

    test_labels = test_labels.flatten()

    count = 0
    for i in range(len(results)):
        if(results[i] == test_labels[i]):
            count += 1

    accuracy = count / len(results)
    print(results)
    print(accuracy * 100)
  
    f,a = plt.subplots(1, 10, figsize = (20, 4))
    j = 0

    for i in range(0, 10):
        print(results[i])
        a[j].imshow(np.reshape(test_data[i], (50, 50)))
        j += 1
