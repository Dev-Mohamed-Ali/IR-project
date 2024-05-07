from flask import Flask, request, jsonify, render_template
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import defaultdict
import re
from Stopwords import all_stop_words
from InverseIndex import InverseIndex
from PositinalIndex import PositionalIndex
from ExtendedBinaryRetrieval import BiwordIndex

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
inverted_index = InverseIndex()
inverted_index.create_posting_list()

# Positional index
positional_index = PositionalIndex()
positional_index.get_posting_list()

# Bi-word index
bi_word_index = BiwordIndex()
bi_word_index.get_posting_list('./Dataset')


# Route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Route for search queries
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').lower()
    index_type = request.args.get('indexType', '')

    # Tokenize the query and remove duplicates
    tokens = list(set(preprocess_document(query)))

    # Initialize result set
    result_docs = set()
    # Bi-word index lookup
    if index_type == 'biWord' and ' ' in query:
        results_bi = bi_word_index.search(query)
        for item in list(set(results_bi)):
            doc_id_text_bi = f"Document ID: {item}"
            # Further processing or actions with the document ID and position can be performed here
            result_docs.add(doc_id_text_bi)
    # Process each token in the query
    for token in tokens:
        # Positional index lookup
        if index_type == 'positional' and token:
            # Search in the positional index
            results = positional_index.lookup(token)[1]
            for inner_doc_id, positions in results.items():
                position_list = ', '.join(str(pos) for pos in positions)
                doc_id_text = f"Document ID: {inner_doc_id}, Positions: [{position_list}]"
                # Further processing or actions with the document ID and position can be performed here
                result_docs.add(doc_id_text)
        # Inverted index lookup
        elif index_type == 'inverted' and token:
            if len(tokens) > 1:
                continue
            # Search in the positional index
            results_inv = inverted_index.lookup(token)
            for item in results_inv:
                doc_id_text_inv = f"Document ID: {item}"
                # Further processing or actions with the document ID and position can be performed here
                result_docs.add(doc_id_text_inv)

    # Prepare response
    response = {
        'query': query,
        'results': list(result_docs)
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
