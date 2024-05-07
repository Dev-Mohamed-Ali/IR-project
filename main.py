from flask import Flask, request, jsonify, render_template
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import defaultdict
import re

app = Flask(__name__)

# Mock data - you should replace this with your actual data collection mechanism
documents = {
    1: "This is the first document. It talks about Python programming.",
    2: "The second document discusses web development with Flask.",
    3: "Python is a popular programming language for data analysis."
}

# Preprocessing functions
def preprocess_document(doc):
    # Tokenization
    tokens = word_tokenize(doc.lower())
    # Stop words removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Inverted index
inverted_index = defaultdict(set)
for doc_id, doc_text in documents.items():
    tokens = preprocess_document(doc_text)
    for token in tokens:
        inverted_index[token].add(doc_id)

# Positional index
positional_index = defaultdict(dict)
for doc_id, doc_text in documents.items():
    tokens = preprocess_document(doc_text)
    for position, token in enumerate(tokens):
        if token not in positional_index:
            positional_index[token] = {}
        if doc_id not in positional_index[token]:
            positional_index[token][doc_id] = []
        positional_index[token][doc_id].append(position)

# Bi-word index
bi_word_index = defaultdict(set)
for doc_id, doc_text in documents.items():
    tokens = preprocess_document(doc_text)
    for i in range(len(tokens)-1):
        bi_word = tokens[i] + ' ' + tokens[i+1]
        bi_word_index[bi_word].add(doc_id)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for search queries
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').lower()
    bi_word = request.args.get('biWord', '') == 'true'
    positional = request.args.get('positional', '') == 'true'
    inverted = request.args.get('inverted', '') == 'true'

    tokens = preprocess_document(query)

    # Initialize result set
    result_docs = set()

    # Process each token in the query
    for token in tokens:
        # Bi-word index lookup
        if bi_word and ' ' in token:
            result_docs.update(bi_word_index.get(token, set()))
        # Positional index lookup
        elif positional and token in positional_index:
            result_docs.update(positional_index[token].keys())
        # Inverted index lookup
        elif inverted and token in inverted_index:
            result_docs.update(inverted_index[token])

    # Prepare response
    response = {
        'query': query,
        'results': list(result_docs)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
