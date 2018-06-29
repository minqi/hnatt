import util.yelp as yelp
from hnatt import HNATT

YELP_DATA_PATH = 'data/yelp-dataset/yelp_review.csv'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'model.h5'
EMBEDDINGS_PATH = 'saved_models/glove.6B.100d.txt'

if __name__ == '__main__':
	(train_x, train_y), (test_x, test_y) = yelp.load_data(path=YELP_DATA_PATH, size=1e4, binary=False)

	# initialize HNATT 
	h = HNATT()	
	h.train(train_x, train_y, 
		batch_size=16,
		epochs=16,
		embeddings_path=None, 
		saved_model_dir=SAVED_MODEL_DIR,
		saved_model_filename=SAVED_MODEL_FILENAME)

	h.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)

	# embeddings = h.word_embeddings(train_x)
	# preds = h.predict(train_x)
	# print(preds)
	# import pdb; pdb.set_trace()

	# print attention activation maps across sentences and words per sentence
	activation_maps = h.activation_maps(
		'they have some pretty interesting things here. i will definitely go back again.')
	print(activation_maps)
