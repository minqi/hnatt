from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify
import tensorflow as tf
from keras import backend as K

from hnatt import HNATT

app = Flask(__name__)

K.clear_session()
h = HNATT()
h.load_weights('saved_models/model.h5')
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
		global graph
		with graph.as_default():
			activation_maps = h.activation_maps(text, as_json=True)
			return jsonify(activation_maps)
	else:
		return Response(status=501)