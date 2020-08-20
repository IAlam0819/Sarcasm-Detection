from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	import os
	import re
	import string
	import nltk
	from nltk.corpus import stopwords
	data = pd.read_csv('sarcasm_data.csv')
	nltk.download('stopwords')
	stopword = nltk.corpus.stopwords.words('english')
	nltk.download('wordnet')
	from nltk.stem import WordNetLemmatizer 
	lemmatizer = WordNetLemmatizer()
	def clean_text(text):
    		text = "".join([word.lower() for word in text if word not in string.punctuation])
    		text = "".join([word for word in text if not word.isdigit()])
    		tokens = re.split('\W+', text)
    		text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopword]
    		return text	
	tfidf_vect = TfidfVectorizer(analyzer=clean_text)
	X_tfidf = tfidf_vect.fit_transform(data['headline'])
	X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
	X_tfidf_df.columns = tfidf_vect.get_feature_names()
	Vectorized_data = X_tfidf_df.iloc[:, 2:]
	x = Vectorized_data.values
	y = data['is_sarcastic'].values
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
	
	clf = MultinomialNB()
	clf.fit(x_train,y_train)
	
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = lemmatizer.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
