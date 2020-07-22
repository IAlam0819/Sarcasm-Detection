#importing pandas and numpy
import pandas as pd
import numpy as np
#importing the data
data = pd.read_csv('sarcasm_data.csv')
data.head()

#removing the first column
data = data.iloc[:,1:]
data.head()

#Shape of the Data
data.shape

#checking the data types
data.dtypes

#checking the missing values
data.isnull().sum()

"""# **Pre-processing text data**"""

#importing important packages
import os
import re
import string
import nltk
from nltk.corpus import stopwords

#checking the unique count
data['is_sarcastic'].value_counts()

#nltk stowords downloading repository
nltk.download('stopwords')

#defined stopwords
stopword = nltk.corpus.stopwords.words('english')
print(stopword)

#wordnet download for lemmatization
nltk.download('wordnet')

#WordNetLemmatizer
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()

#cleaning function
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    text = "".join([word for word in text if not word.isdigit()])
    tokens = re.split('\W+', text)
    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopword]
    return text

#cleaning the headline
list_of_words = clean_text(data['headline'])

#finding the total number of words after pre-processing of the text
print("length of the list of words", len(list_of_words))
print(list_of_words)

#counting the frequency of each word in list of words
counts = dict()
for word in list_of_words:
  counts[word] = counts.get(word, 0) + 1
print(counts)

#sorting the dictionary in decresing order
sorted_counts = sorted(counts.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
print(sorted_counts)

#storing only the keys of the dictionary in the list
words = list()
for w in sorted_counts:
  words.append(w[0])
print(words)

#subsetting the list, to find the top 200 most frequent used words
top_200_words = words[0:200]
print(top_200_words)

#coverting the list into string
comment_words = '' 
comment_words += " ".join(top_200_words)+" "
print(comment_words)

#generating the wordcloud for 200 most frequent used words
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
wordcloud = WordCloud(width = 800, height = 700, 
                background_color ='black',  
                min_font_size = 10).generate(comment_words) 
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show()

"""# **Vectorizing Raw Data: TF-IDF**

TF-IDF Creates a document-term matrix where the columns represent single unique terms (unigrams) but the cell represents a weighting meant to represent how important a word is to a document.
"""

#Term Document Matrix using tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer=clean_text)

#passing the headline column of data
X_tfidf = tfidf_vect.fit_transform(data['headline'])

#shape of the matrix
print(X_tfidf.shape)
#feature names
print(tfidf_vect.get_feature_names())

#converting matrix into a dataframe
X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
X_tfidf_df.columns = tfidf_vect.get_feature_names()

X_tfidf_df.head()

#final vectorized data
Vectorized_data = X_tfidf_df.iloc[:, 2:]
Vectorized_data.head()

"""# **Building the Model**
we will be building the model using different ML algorithms, starting with multiple linear regression and logistic regression, followed by KNN, naive bayes, decision tree and random forest.
"""

# independant variables
x = Vectorized_data.values
# the dependent variable
y = data['is_sarcastic'].values

#shape of the dependent and independent data
print("Shape of Independent data ", x.shape)
print("Shape of Dependent data ",y.shape)

# Split X and y into training and test set in 70:30 ratio
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

#Shape
print("Training data :")
print(x_train.shape)
print(y_train.shape)
print("Testing data :")
print(x_test.shape)
print(y_test.shape)

"""# **Logistic Regression**"""

#importing LogisticRegression and accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#defining the model
LogRes_model = LogisticRegression()
#fitting the train set to the model
LogRes_model.fit(x_train, y_train)

#prediction made
prediction = LogRes_model.predict(x_test)
print(prediction)

#confusion_matrix
conf_matrix = confusion_matrix(y_test, prediction)
print(conf_matrix)

#accuracy_score of the model
accuracy = accuracy_score(y_test, prediction)
print(accuracy)

#classification model
from sklearn.metrics import classification_report
report = classification_report(y_test, prediction)
print(report)

"""# **Naive Bayes Classifier**"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report
#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(x_train,y_train)

#Naive Bayes predictions
y_pred = clf.predict(x_test)

#Naive Bayes confusion matrix
NB_conf_matrix = confusion_matrix(y_test, y_pred)
print(NB_conf_matrix)

#Naive Bayes accuracy
NB_accuracy = accuracy_score(y_test, y_pred)
print(NB_accuracy)

#Naive Bayes classification report
print(classification_report(y_test, y_pred))

#ROC Curve
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)
#plotting the ROC curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Plot the Precision-Recall curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(clf, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))

