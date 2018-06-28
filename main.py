import yelp
from hnatt import HNATT

YELP_DATA_PATH = 'data/yelp-dataset/yelp_review.csv'

if __name__ == '__main__':
	# (train_x, train_y), (test_x, test_y) = yelp.load_data(path=YELP_DATA_PATH, size=1e5)

	# initialize HNATT 
	h = HNATT()
	# h.train(train_x, train_y, checkpoint_path='saved_models/model.h5')
	h.load_weights('saved_models/model.h5')

	# print attention activation maps across sentences and words per sentence
	activation_maps = h.activation_maps(
		'they have some pretty interesting things here. i will definitely go back again.')
	print(activation_maps)