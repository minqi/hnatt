import pandas as pd
import numpy as np
from tqdm import tqdm

from util.text_util import normalize

tqdm.pandas()

def chunk_to_arrays(chunk, binary=False):
	x = chunk['text_tokens'].values
	if binary:
		y = chunk['polarized_stars'].values
	else:
		y = chunk['stars'].values
	return x, y

def balance_classes(x, y, dim, train_ratio):
	x_negative = x[np.where(y == 1)]
	y_negative = y[np.where(y == 1)]
	x_positive = x[np.where(y == 2)]
	y_positive = y[np.where(y == 2)]

	n = min(len(x_negative), len(x_positive))
	train_n = int(round(train_ratio * n))
	train_x = np.concatenate((x_negative[:train_n], x_positive[:train_n]), axis=0)
	train_y = np.concatenate((y_negative[:train_n], y_positive[:train_n]), axis=0)
	test_x = np.concatenate((x_negative[train_n:], x_positive[train_n:]), axis=0)
	test_y = np.concatenate((y_negative[train_n:], y_positive[train_n:]), axis=0)

	# import pdb; pdb.set_trace()
	return (train_x, to_one_hot(train_y, dim=2)), (test_x, to_one_hot(test_y, dim=2))

def to_one_hot(labels, dim=5):
	results = np.zeros((len(labels), dim))
	for i, label in enumerate(labels):
		results[i][label - 1] = 1
	return results

def polarize(v):
	if v >= 3:
		return 2
	else:
		return 1

def load_data(path, size=1e4, train_ratio=0.8, binary=False):
	print('loading Yelp reviews...')
	df = pd.read_csv(path, nrows=size, usecols=['stars', 'text'])
	df['text_tokens'] = df['text'].progress_apply(lambda x: normalize(x))
	
	dim = 5
	if binary:
		dim = 2

	if binary:
		df['polarized_stars'] = df['stars'].apply(lambda x: polarize(x))
		x, y = chunk_to_arrays(df, binary=binary)
		return balance_classes(x, y, dim, train_ratio)

	train_size = round(size * train_ratio)
	test_size = size - train_size;

	# training + validation set
	train_x = np.empty((0,))
	train_y = np.empty((0,))

	train_set = df[0:train_size].copy()
	train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))
	# train_set.sort_values('len', inplace=True, ascending=True)
	train_x, train_y = chunk_to_arrays(train_set, binary=binary)
	train_y = to_one_hot(train_y, dim=dim)

	test_set = df[train_size:]
	test_x, test_y = chunk_to_arrays(test_set, binary=binary)
	test_y = to_one_hot(test_y)
	print('finished loading Yelp reviews')

	return (train_x, train_y), (test_x, test_y)
