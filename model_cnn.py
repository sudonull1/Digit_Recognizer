# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:48:22 2020

@author: SAURABH SINGH
"""


#mnist loader 
import numpy as np
import tensorflow as tf
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#progress bar
import sys

def progress(count, total, cond=False):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '-' * (bar_len - filled_len)

    if cond == False:
    	sys.stdout.write('[%s] %s%s\r' % (bar, percents, '%'))
    	sys.stdout.flush()

    else:
    	sys.stdout.write('[%s] %s%s' % (bar, percents, '%'))
#lines 34-39 are used exclusively in Colab Notebook
from google.colab import files
uploaded = files.upload()

for fn in uploaded.keys():
 print('User uploaded file "{name}" with length {length} bytes'.format(
     name=fn, length=len(uploaded[fn])))
        
#CNN structure 
import random
import gzip
import pickle

f = gzip.open('mnist.pkl.gz', 'rb')

training_data, validation_data, test_data = pickle.load(f, encoding = 'latin1')

f.close()


tr_inputs = [np.reshape(x, (1,784)) for x in training_data[0]]
tr_outputs = [vectorized_result(x) for x in training_data[1]]


te_inputs = [np.reshape(x, (1,784)) for x in test_data[0]]
te_outputs = [vectorized_result(x) for x in test_data[1]]


#we need to classify 0-9 (10 class)

n_class = 10
batch_size = 256

import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()


x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# x = tf.compat.v1.placeholder(tf.float32, shape = (None, 784))
# y = tf.compat.v1.placeholder(tf.float32)

#convolution network structure

def conv_net(x):
  weights = {'W_conv1' : tf.Variable(tf.random_normal([5,5,1,32])),
			          'W_conv2' : tf.Variable(tf.random_normal([5,5,32,64])),
			          'W_fc' : tf.Variable(tf.random_normal([7*7*64,1024])),
			          'out' : tf.Variable(tf.random_normal([1024,n_class])),
			   }
  biases = { 'b_conv1' : tf.Variable(tf.random_normal([32])),
			        'b_conv2' : tf.Variable(tf.random_normal([64])),
			        'b_fc' : tf.Variable(tf.random_normal([1024])),
			        'out' : tf.Variable(tf.random_normal([n_class])),
			  }		  
        
  x = tf.reshape(x, shape = [-1, 28, 28, 1])
  
  conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
  
  conv1 = maxpool2d(conv1)
  
  conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
  
  conv2 = maxpool2d(conv2)
  
  fc = tf.reshape(conv2, [-1,7*7*64])
  fc = tf.nn.relu(tf.matmul(fc,weights['W_fc']) + biases['b_fc'])
  
  ouptut = tf.matmul(fc, weights['out']) + biases['out']
  
  return ouptut


#training network  

def train_net(x):

	sess = tf.InteractiveSession
	prediction = conv_net(x)
	print(prediction)
	cost = tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels=y)
	optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)
	correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess :
		sess.run(tf.global_variables_initializer())

		for i in range(20000):

			if(i % 200) == 0 :
				progress(i, 20000)

			for j in range(10):
				k = random.randint(0,20000)
				optimizer.run(feed_dict = {x: tr_inputs[k], y: tr_outputs[k]})

		progress(20000, 20000, cond = True)

		print("\n")

		acc = 0
		for i in range(10000):
			acc += accuracy.eval(feed_dict={
				x: te_inputs[i], y: te_outputs[i]})

		print('test accuracy %g' % acc)




train_net(x)
