import rbe
import os
import time
import flask
import pickle
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

global stemmer
global tokenizer
global dictionary
global modified_files
stemmer = SnowballStemmer('english')
tokenizer = TweetTokenizer()
app = flask.Flask(__name__)
	
try:
	with open('dictionary.pk', 'rb') as f:
		dictionary = pickle.load(f)
except:
	dictionary = pd.DataFrame()

try:
	with open('modified_files.pk', 'rb') as f:
		modified_files = pickle.load(f)
except:
	modified_files = dict()

def save_dictionary():
	global dictionary
	with open('dictionary.pk', 'wb') as f:
		pickle.dump(dictionary, f)

def save_modifiedlist():
	global modified_files
	with open('modified_files.pk', 'wb') as f:
		pickle.dump(modified_files, f)

def get_tokens(fname):
	with open('documents/%s' % fname, 'r') as f:
		text = f.read()
	text = re.sub('\n,()}{/\\*&^%$#@!', ' ', text)
	tokens = tokenizer.tokenize(text)
	normalized_tokens = [stemmer.stem(token) for token in tokens]
	token_counts = pd.value_counts(normalized_tokens)	
	return token_counts

def append_token(token_counts, docname):
	global dictionary
	dictionary[docname] = 0
	for index in token_counts.index:
		if index in dictionary.index:
			dictionary.loc[index, docname] = token_counts[index]
		else:
			dictionary.loc[index] = 0
			dictionary.loc[index, docname] = token_counts[index]


def update_db():
	global dictionary 
	global tokenizer
	global modified_files

	while True:
		is_modified = False
		files = os.listdir('documents')
		modified_time = {file: os.path.getatime('documents/%s'%file) for file in files}
		for file in files:
			if file not in modified_files:
				token_counts = get_tokens(file)
				append_token(token_counts, file)
				modified_files[file] = modified_time[file]
				is_modified = True
			elif modified_files[file] != modified_time[file]:
				dictionary[file] = 0
				token_counts = get_tokens(file)
				append_token(token_counts, file)
				modified_files[file] = modified_time[file]
				is_modified = True
			else:
				pass

		if is_modified is True:
			dictionary.sort_index(inplace=True)
			save_modifiedlist()
			save_dictionary()
		time.sleep(15)


@app.route('/')
def search_query():
	global stemmer
	global dictionary

	query = flask.request.args.get('q')
	if query is None:
		return flask.render_template('home.html')
	tokens = re.split('\s+', query)
	normalized_tokens = np.unique([stemmer.stem(token) for token in tokens])
	doc_len = len(dictionary.columns)
	token_len = len(normalized_tokens)
	cal_idf= lambda x: np.log10(doc_len/sum(dictionary.loc[x]>0)) if x in dictionary.index else 1
	tokens_idf= {token: cal_idf(token) for token in normalized_tokens}
	doc_scores = pd.DataFrame(np.zeros((token_len, doc_len)), index=list(normalized_tokens), columns=list(dictionary.columns))

	for doc in dictionary.columns:
		for token in normalized_tokens:
			tf = dictionary.loc[token, doc] if token in dictionary.index else 0
			doc_scores.loc[token, doc] = np.log10(1+tf) * tokens_idf[token]
	doc_scores = doc_scores.sum(axis=0)
	doc_scores = doc_scores[doc_scores > 0]
	doc_scores.sort_values(inplace=True, ascending=False)
	results = {}
	for file in doc_scores.index:
		with open('documents/%s' % file) as f:
			head = [f.readline() for _ in range(10)]
		results[file] = ''.join(head).strip()
	return flask.render_template('result.html', length=len(results), results=results)

@app.route('/doc/<int:docnum>')
def show_doc(docnum):
	documents = os.listdir('documents')
	if str(docnum) in documents:
		with open('documents/%d' % docnum, 'r') as f:
			text = f.read()
		return flask.Response(text)
	else:
		return flask.abort(404) 
		
if __name__ == '__main__':
	updating_thread = threading.Thread(target=update_db, daemon=True)
	updating_thread.start()
	app.run("127.0.0.1", 8000)