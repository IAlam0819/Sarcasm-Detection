# SARCASM-DETECTION
Sarcasm is the use of language that normally signifies the opposite in order to mock or convey
contempt. Sarcasm detection is a very narrow research field in NLP, a specific case of sentiment
analysis where instead of detecting a sentiment in the whole spectrum, the focus is on sarcasm.
Therefore, the task of this field is to detect if a given text is sarcastic or not.

# Dataset
The News Headlines Dataset for Sarcasm Detection is collected from two news website. And
their news headlines from different Post.
Number of attributes:
Each record consists of three attributes:
1. is_sarcastic: 1 if the record is sarcastic otherwise 0.
2. headline: the headline of the news article.
3. article_link: link to the original news article.

# Pre-processing
For starting working on the project, we will start with cleaning the data or pre-processing the
data, which is followed by the vectorising the data. These steps are also referred to as Text
classification. 
For text classification, we need to follow two basic steps:

1. A pre-processing step to make the texts cleaner and easier to process.
2. And a vectorization step to transform these texts into numerical vectors.
Pre-processing steps includes the followings:

• Punctuation marks
• Numbers
• Tokenization
• Stop words
• Stemming/Lemmatization

# Word Embedding
Word embedding is the collective name for a set of language modelling and feature
learning techniques in natural language processing (NLP) where words or phrases from
the vocabulary are mapped to vectors of real numbers. There are following word
embedding techniques:

Count Vectorizer
TF-IDF Vectorizer
Hashing Vectorizer
Word2Vec

We will be using TF-IDF Vectorizer for word embedding of our data. TF-IDF stands for “Term
Frequency — Inverse Document Frequency”. This is a technique to quantify a word in
documents, we generally compute a weight to each word which signifies the importance of the
word in the document and corpus.

# Building the prediction model
we used logistic regression and Multinomial naive bayes for predicting the data.
Multinomial naive bayes showed better results and accuracy of 79%.
