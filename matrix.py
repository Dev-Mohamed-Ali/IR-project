import os
import re

import numpy as np
import nltk
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

from Lemmatizer import lemmatize_text
from Stopwords import all_stop_words

# Directory containing the text files
directory = 'Dataset'

# Initialize list to store document content
documents = []


def preprocess_document(doc):
    query = re.sub(r'[^\w\s]', '', doc)
    query = re.sub(r'\d+', '', query)
    query = re.sub(r'\s+', ' ', query)
    query = query.strip()
    query = lemmatize_text(query)[0]

    stopwords_all = all_stop_words()
    # Stop words removal
    tokens = [token for token in query if token not in stopwords_all]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


# Iterate over each file in the directory
for filename in os.listdir(directory)[0:3]:
    if filename.endswith(".txt"):
        # Read the contents of the file
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            fix = preprocess_document(content)
            # Append the content to the documents list
            documents.append(" ".join(fix))
# Tokenize the documents
tokens = []
for doc in documents:
    tokens.extend(word_tokenize(doc.lower()))

# Remove punctuation and stopwords
stop_words = set(stopwords.words('english'))
tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]

# Get unique terms
terms = list(set(tokens))

# Initialize the term-document matrix
term_doc_matrix = np.zeros((len(terms), len(documents)), dtype=int)

# Populate the term-document matrix
for i, term in enumerate(terms):
    for j, document in enumerate(documents):
        if term in document.lower():
            term_doc_matrix[i][j] = 1

# Print the term-document matrix
print("Term-Document Incidence Matrix:")
print("---------------------------------")
print("      ", end="")
for j in range(len(documents)):
    print(f"Document {j + 1} |", end=" ")
print()
print("---------------------------------")
for i, term in enumerate(terms):
    print(f"{term:<6}", end=" | ")
    for j in range(len(documents)):
        print(f"{term_doc_matrix[i][j]:^10}", end=" | ")
    print()
