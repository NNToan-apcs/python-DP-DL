from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
export="C:\\Users\\nntoa\\AppData\\Local\\Temp\\tmp0manor2p\\model.ckpt-3510"
    
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export)