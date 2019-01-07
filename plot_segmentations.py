# from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import numpy,Image
from random import random
import sys,skimage
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import os
import glob
# image_path=sys.argv[1]
# img = img_as_float(astronaut()[::2, ::2])
files=glob.glob("/home/user/project/demo/pic/*.jpg")
for image_path in files:
	img = skimage.io.imread(image_path)
	segments_fz = felzenszwalb(img, scale=600, sigma=0.8, min_size=100)

	gradient = sobel(rgb2gray(img))
	inp_image = numpy.append(
	    img, numpy.zeros(img.shape[:2])[:, :, numpy.newaxis], axis=2)
	inp_image[:, :, 3] = segments_fz
	# print inp_image
	im=Image.open(image_path)
	width,height=im.size
	print width,height




	# print segments_fz

	print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))

	n=len(np.unique(segments_fz))

	random_color = lambda: (int(random()*255), int(random()*255), int(random()*255))
	colors = [random_color() for i in xrange(n)]
	# print colors

	from PIL import Image
	imc = Image.open(image_path)
	pixelMap = imc.load()

	for i in range(height):
		for j in range(width):
			pixelMap[j,i]=colors[segments_fz[i][j]]
			# print segments	_fz[i][j]
	# imc.show()
	head,tail = os.path.split(image_path)
	path1="/home/user/project/demo/segments/"+tail
	imc.save(path1)

	# print len(segments_fz)

	# fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True,subplot_kw={'adjustable': 'box-forced'})

	# ax[0, 0].imshow(mark_boundaries(img, segments_fz))
	# # print segments_fz
	# ax[0, 0].set_title("Felzenszwalbs's method")


	# for a in ax.ravel():
	#     a.set_axis_off()

	# plt.tight_layout()
	# plt.show()
