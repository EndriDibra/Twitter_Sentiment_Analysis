# Author: Endri Dibra

# importing necessary libraries
import re
import string
import numpy as np
import pandas as pd

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

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
stop_words_set = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above",
                  "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added",
                  "adj","ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against",
                  "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already",
                  "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce",
                  "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere",
                  "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are",
                  "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at",
                  "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba",
                  "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before",
                  "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below",
                  "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl",
                  "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1",
                  "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce",
                  "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon",
                  "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering",
                  "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course",
                  "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da",
                  "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did",
                  "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don",
                  "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e",
                  "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either",
                  "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo",
                  "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]


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
    group_names = ['True Neg ','False Pos ', 'False Neg ','True Pos ']

    group_percentages = ['{0:.2%}'.format(value) for value in matrix.flatten() / np.sum(matrix)]

    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(matrix, annot=labels, cmap='Blues', fmt='', xticklabels=categories, yticklabels=categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Matrix", fontdict = {'size':18}, pad = 20)


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
plt.legend(loc="lower right")
plt.show()
