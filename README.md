# HNATT
This is a Keras implementation of the **H**ierarchical **N**etwork with **Att**ention architecture [(Yang et al, 2016)](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf). 

### Overview
HNATT is a deep neural network for document classification. It learns hierarchical hidden representations of documents at word, sentence, and document levels. At both the word and sentence levels, HNATT makes use of an attention mechanism, in which it learns a context vector that determines a relevance weighting for its learned encoding of words and sentences. 

### Contents
| Module | Description |
| ------ | ------ |
| `hnatt.py`* | Main HNATT implementation with custom Attention layer. |
| `yelp.py` | Data loader for Yelp review data used for training and testing. |
| `text_util.py` | Utility function for normalizing review texts. |
| `main.py` | Demo that trains HNATT on a subset of Yelp reviews and displays attention activation maps at both sentence and word levels on an example review. |
*A TensorFlow backend is assumed by the Attention layer.

### Get started
Install dependencies in a new virtual environement via
```
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Give it a spin
```
python main.py
```

Load `n` reviews from yelp for training, with 90/10 training/test split:
```
import yelp
(train_x, train_y), (test_x, test_y) = yelp.load_data(path=YELP_DATA_PATH, size=1e5, train_ratio=0.9)
```

Train your HNATT
```
h = HNATT()
h.train(train_x, train_y, checkpoint_path='saved_models/model.h5')
```

View sentence and word-level attention activations
```
activation_maps = h.activation_maps('they have some pretty interesting things here. i will definitely go back again.')
print(activation_maps)
```