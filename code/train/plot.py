import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import sys
import os.path as osp

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush(save_path):
	prints = []

	for name, vals in _since_last_flush.items():
		prints.append("{}\t{}".format(name, np.mean( list(vals.values())  )))
		_since_beginning[name].update(vals)

		x_vals = np.sort(  list(_since_beginning[name].keys())  )
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig( osp.join(save_path, name.replace(' ', '_')+'.jpg')  )

	print ("iter {}\t{}".format(_iter[0], "\t".join(prints)))
	_since_last_flush.clear()
