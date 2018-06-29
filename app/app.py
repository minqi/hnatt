import numpy as np

from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify
import tensorflow as tf
from keras import backend as K

from util import text_util
from hnatt import HNATT

SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'model.h5'

app = Flask(__name__)

K.clear_session()
h = HNATT()
h.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
graph = tf.get_default_graph()

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/activations')
def activations():
	"""
	Receive a text and return HNATT activation map
	"""
	if request.method == 'GET':
		text = request.args.get('text', '')
		if len(text.strip()) == 0:
			return Response(status=400)

		ntext = text_util.normalize(text)

		global graph
		with graph.as_default():
			activation_maps = h.activation_maps(text, websafe=True)
			preds = h.predict([ntext])[0]
			prediction = np.argmax(preds).astype(float)
			data = {
				'activations': activation_maps,
				'normalizedText': ntext,
				'prediction': prediction,
				'binary': preds.shape[0] == 2
			}
			return jsonify(data)
	else:
		return Response(status=501)