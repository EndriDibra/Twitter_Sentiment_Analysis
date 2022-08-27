# Author: Endri Dibra

# importing necessary libraries
import re
import string
import numpy as np
import pandas as pd

# plotting libraries (seaborn and matplotlib)
import seaborn as sns
import matplotlib.pyplot as plt

# nltk library
import nltk
from nltk.tokenize import RegexpTokenizer

# sklearn library
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


# Loading and reading  dataset that will be used in this programme
dataset_columns = ['Target', 'IDs', 'Date', 'Flag', 'User', 'Text']
dataset_encoding = "ISO-8859-1"
dataset = pd.read_csv('Twitter_Dataset.csv', encoding=dataset_encoding, names=dataset_columns)

# exploratory phase
# getting the five top records of dataset
print(dataset.head())

# printing columns of dataset
print("The columns of dataset are below :")
print(dataset.columns)

# printing the length of rows and columns
print("The length of rows is :", len(dataset))
print("The length of columns is :", len(dataset.columns))

# printing the length of dataset
print("Length of dataset is :", len(dataset))

# printing the size of dataset
print("Dataset's size is :", dataset.shape)

# printing the information of dataset
print("Info of dataset below :")
print(dataset.info())

# printing the datatypes of all columns of dataset
print(dataset.dtypes)

# checking for null values (non-existing) in dataset
print("Null values :", np.sum(dataset.isnull().any(axis=1)))

# checking unique target values in dataset
print("Unique target values :", dataset['Target'].unique())

# checking the number of target values in dataset
print("# of target values :", dataset['Target'].nunique())

# data preprocessing phase
# choosing the text and target column to analyze
data = dataset[['Text','Target']]

# replacing value 4 with value 1, for a correct sentiment analysis
data['Target'] = data['Target'].replace(4,1)

# unique values of target variables
data['Target'].unique()

# distinguishing negative and positive tweets of the dataset
neg_tw = data[data['Target'] == 0]
pos_tw = data[data['Target'] == 1]

# taking and using 25% of our data
neg_tw = neg_tw.iloc[:int(20000)]
pos_tw = pos_tw.iloc[:int(20000)]

# combining together negative and positive data (tweets)
tweets = pd.concat([neg_tw,pos_tw])

# converting data (text) in lowercase
tweets['Text'] = tweets['Text'].str.lower()
tweets['Text'].tail()

# creating a set of stopwords to clean and improve the dataset
stop_words_set = open("stop_words.txt").read().lower()

# cleaning and removing all those words from the dataset
Stop_Words = set(stop_words_set)


#creating function to remove stopwords
def cleaning_stopwords(text):

    return " ".join([word for word in str(text).split() if word not in Stop_Words])

tweets['Text'] = tweets['Text'].apply(lambda text: cleaning_stopwords(text))
tweets['Text'].head()


# cleaning and removing punctuations
english_punctuations = string.punctuation
punctuations_set = english_punctuations

def cleaning_punctuations(text):

    translator = str.maketrans('', '', punctuations_set)
    return text.translate(translator)

tweets['Text']= dataset['Text'].apply(lambda x: cleaning_punctuations(x))
tweets['Text'].tail()


# cleaning and removing repeating characters
def cleaning_repeating_characters(text):

    return re.sub(r'(.)1+', r'1', text)

tweets['Text'] = tweets['Text'].apply(lambda x: cleaning_repeating_characters(x))
tweets['Text'].tail()


# cleaning and removing URL addresses
def cleaing_URL_Addresses(data):

    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ', data)

tweets['Text'] = tweets['Text'].apply(lambda x: cleaing_URL_Addresses(x))
tweets['Text'].tail()


# cleaning and removing numeric numbers
def cleaning_numbers(data):

    return re.sub('[0-9]+', '', data)

tweets['Text'] = tweets['Text'].apply(lambda x: cleaning_numbers(x))
tweets['Text'].tail()

# getting tokenization of tweet text
tokenizer = RegexpTokenizer(r'w+')
tweets['Text'] = tweets['Text'].apply(tokenizer.tokenize)
tweets['Text'].head()

# applying stemming
stemming = nltk.PorterStemmer()


def stemming_on_dataset(data):

    text = [stemming.stem(word) for word in data]
    return data

tweets['Text']= tweets['Text'].apply(lambda x: stemming_on_dataset(x))
tweets['Text'].head()

# applying lemmatizer
lemmatizer = nltk.WordNetLemmatizer()


def lemmatizer_on_dataset(data):

    text = [lemmatizer.lemmatize(word) for word in data]
    return data

tweets['Text'] = tweets['Text'].apply(lambda x: lemmatizer_on_dataset(x))
tweets['Text'].head()

# Separating input feature and label
X = data.Text
y = data.Target

# Separating the 95% data for training data and 5% for testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)

# Fitting the TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(X_train)
print('Number of feature words: ', len(vectorizer.get_feature_names()))

# transforming the data using TF-IDF vectorizer
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)


# evaluating the model
def model_Evaluate(model):

    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))

    # Compute and plot the Confusion matrix
    matrix = confusion_matrix(y_test, y_pred)

    categories = ['Negative','Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    group_percentages = ['{0:.2%}'.format(value) for value in matrix.flatten() / np.sum(matrix)]

    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(matrix, annot=labels, cmap='Reds', fmt='', xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict = {'size':16}, labelpad = 10)
    plt.ylabel("Values" , fontdict = {'size':16}, labelpad = 10)
    plt.title ("Matrix", fontdict = {'size':22}, pad = 20)


# building model "Logistic Regression"
LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)


# plotting Logistic Regression model
a,b,thresholds = roc_curve(y_test, y_pred3)
roc_auc = auc(a,b)
plt.figure()
plt.plot(a, b, color='darkblue', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(title="User's texts sentiment")
plt.show()
