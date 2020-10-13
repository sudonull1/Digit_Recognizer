#mnist loader 
import pickle
import gzip

import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding = 'utf8')
    f.close()
    return (training_data, validation_data, test_data)

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


#network params

def sigmoid(z):
  return 1/(1+np.exp(-z))

def del_sig(z):
  return z*(1-z)


def training(inp, out, weight, weight1):
  x = inp.T
  y = out.T

  l1 = sigmoid(np.dot(x, weight))
  l2 = sigmoid(np.dot(l1, weight1))

  error = y-l2
  l2_del = error* del_sig(l2)
  e0 = l2_del.dot(weights.T)
  l1_del = e0 * del_sig(l1)

  #updating the values for how much we missed 
  
  weight1 += np.dot(l1.T, l2_del)
  weight += np.dot(x.T, l1_del)
  return weight, weight1


#this is for creating test accuracy

def feed_fore(x, weight, weight1):
  l = x.T
  l1 = sigmoid(np.dot(l,weight));
  l2 = sigmoid(np.dot(l, wegiht1));
  return l2;

def check_bar(in_vec, out_vec, weight, weight1):
  correct = 0;
  for i in range(len(in_vec)):
    out = feed_fore(in_vec[i], weight, weight1)
    f_out = np.argmax(out)
    if(f_out == out_vec[i]):
      correct += 1
  print("accuracy = ", ((correct/10000)*100))
  
  
