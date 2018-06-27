import pandas as pd
import numpy as np
from tqdm import tqdm

from text_util import normalize

tqdm.pandas()

def chunk_to_arrays(chunk):
	x = chunk['text_tokens'].values
	y = chunk['stars'].values
	return x, y

def load_data(path, size=1e4, train_ratio=0.9):
	print('loading Yelp reviews...')
	train_size = round(size * train_ratio)
	test_size = size - train_size;

	# training + validation set
	train_x = np.empty((0,))
	train_y = np.empty((0,))
	df = pd.read_csv(path, nrows=size, usecols=['stars', 'text'])
	df['text_tokens'] = df['text'].progress_apply(lambda x: normalize(x))

	train_set = df[0:train_size].copy()
	train_set['len'] = train_set['text_tokens'].apply(lambda x: len(x))
	train_set.sort_values('len', inplace=True, ascending=True)
	train_x, train_y = chunk_to_arrays(train_set)

	test_set = df[train_size:]
	test_x, test_y = chunk_to_arrays(test_set)
	print('finished loading Yelp reviews')

	return (train_x, train_y), (test_x, test_y)
