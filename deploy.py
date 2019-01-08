
#####    using custom 279 dataset
#####    usage : python deploy.py image.jpg


import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib

import tensorflow as tf
image_path=sys.argv[1]

image_data = tf.gfile.FastGFile(image_path, 'rb').read()
label_lines=[line.rstrip() for line in tf.gfile.GFile("/pathproject/tensor/output_labels.txt")]


with tf.gfile.FastGFile("/pathproject/tensor/output_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')


with tf.Session() as sess:
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	predictions = sess.run(softmax_tensor,
	                       {'DecodeJpeg/contents:0': image_data})

	top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
	x=0
	for node_id in top_k:
	  human_string = label_lines[node_id]
	  score = predictions[0][node_id]
	  print('%s (score = %.5f)' % (human_string, score))
	  if x==5:
	  	break
	  else:
	  	x+=1

