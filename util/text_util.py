import string
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

STOP_WORDS = ['the', 'a', 'an']
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

def normalize(text):
	text = text.lower().strip()
	doc = nlp(text)
	filtered_sentences = []
	for sentence in doc.sents:
		filtered_tokens = list()
		for i, w in enumerate(sentence):
			s = w.string.strip()
			if len(s) == 0 or s in string.punctuation and i < len(doc) - 1:
				continue
			if s not in STOP_WORDS:
				s = s.replace(',', '.')
				filtered_tokens.append(s)
		filtered_sentences.append(' '.join(filtered_tokens))
	return filtered_sentences