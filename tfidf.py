# I used Python 3.8 for this HW
import pandas as pd
import numpy as np
#import nltk
#nltk.download('wordnet')
import nltk
import ssl
import sklearn
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import contractions

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("wordnet")

import re
from bs4 import BeautifulSoup

raw_data = pd.read_csv('data.tsv',sep="\t", usecols = ['star_rating', 'review_headline', 'review_body'])

#drop any row that contains NA
raw_data_1 = raw_data.dropna(axis = 0, how = 'any')

raw_data_2 = raw_data_1.copy()
raw_data_2['star_rating'] = raw_data_2.star_rating.astype(str)

data_dict = {}
data_dict[1] = raw_data_2.loc[(raw_data_2['star_rating'] == '1') | (raw_data_2['star_rating'] == '2')].sample(n = 20020, replace = False)
data_dict[2] = raw_data_2.loc[raw_data_2['star_rating'] == '3'].sample(n = 20020, replace = False)
data_dict[3] = raw_data_2.loc[(raw_data_2['star_rating'] == '4') | (raw_data_2['star_rating'] == '5')].sample(n = 20020, replace = False)

ordered_data = pd.concat([data_dict[1], data_dict[2], data_dict[3]])
ordered_data['star_rating'] = ordered_data.star_rating.astype(int)
data_set = sklearn.utils.shuffle(ordered_data)

# compute average length
avg_len_before_cleaning = data_set.review_headline.str.len().mean() + data_set.review_body.str.len().mean()

#convert to lower case
data_set_1 = data_set.copy()
data_set_1['review_headline'] = data_set_1['review_headline'].str.lower()
data_set_1['review_body'] = data_set_1['review_body'].str.lower()

# remove html tags
data_set_2 = data_set_1.copy()
data_set_2['review_headline'] = data_set_2['review_headline'].replace(r"<.*?>", " ", regex=True)
data_set_2['review_body'] = data_set_2['review_body'].replace(r"<.*?>", " ", regex=True)
'''
def handle_html(text):
    html_pattern = re.compile('</?\w+[^>]*>', re.S)
    text = re.sub(html_pattern, '', text)
    return text
data_set_2['review_body'] = data_set_2['review_body'].apply(handle_html)
data_set_2['review_headline'] = data_set_2['review_headline'].apply(handle_html)
'''

# remove urls
data_set_3 = data_set_2.copy()
data_set_3['review_body'] = data_set_3['review_body'].replace(r'http*\S+|www*\S+', ' ', regex=True)
data_set_3['review_headline'] = data_set_3['review_headline'].replace(r'http*\S+|www*\S+', ' ', regex=True)

def do_contractions(text: str):
    result = []
    for word in text.split():
        result.append(contractions.fix(word))
    return " ".join(result)
data_set_4 = data_set_3.copy()
data_set_4['review_body'] = data_set_4['review_body'].apply(do_contractions)
data_set_4['review_headline'] = data_set_4['review_headline'].apply(do_contractions)

data_set_4['review_body'] = data_set_4['review_body'].replace(r'dont', 'do not', regex=True)
data_set_4['review_headline'] = data_set_4['review_headline'].replace(r'dont', 'do not', regex=True)

data_set_4['review_body'] = data_set_4['review_body'].replace(r'wont', 'will not', regex=True)
data_set_4['review_headline'] = data_set_4['review_headline'].replace(r'wont', 'will not', regex=True)

data_set_4['review_body'] = data_set_4['review_body'].replace(r'cant', 'can not', regex=True)
data_set_4['review_headline'] = data_set_4['review_headline'].replace(r'cant', 'can not', regex=True)

data_set_4['review_body'] = data_set_4['review_body'].replace(r'wouldnt', 'would not', regex=True)
data_set_4['review_headline'] = data_set_4['review_headline'].replace(r'wouldnt', 'would not', regex=True)

data_set_4['review_body'] = data_set_4['review_body'].replace(r'didnt', 'did not', regex=True)
data_set_4['review_headline'] = data_set_4['review_headline'].replace(r'didnt', 'did not', regex=True)

data_set_4['review_body'] = data_set_4['review_body'].replace(r'doesnt', 'does not', regex=True)
data_set_4['review_headline'] = data_set_4['review_headline'].replace(r'doesnt', 'does not', regex=True)

data_set_4['review_body'] = data_set_4['review_body'].replace(r'll', 'will', regex=True)
data_set_4['review_headline'] = data_set_4['review_headline'].replace(r'll', 'will', regex=True)

# remove non-alphbet chars
data_set_5 = data_set_4.copy()
data_set_5['review_body'] = data_set_5['review_body'].replace(r"[^0-9a-zA-Z ]", ' ', regex=True)
data_set_5['review_headline'] = data_set_5['review_headline'].replace(r"[^0-9a-zA-Z ]", ' ', regex=True)

# remove extra whitespaces
data_set_6 = data_set_5.copy()
data_set_6['review_body'] = data_set_6['review_body'].replace(r"\s+", ' ', regex=True)
data_set_6['review_headline'] = data_set_6['review_headline'].replace(r"\s+", ' ', regex=True)

avg_len_after_cleaning = data_set_6.review_headline.str.len().mean() + data_set_6.review_body.str.len().mean()
print(avg_len_before_cleaning.astype(str) + ', ' + avg_len_after_cleaning.astype(str))

avg_len_before_preprocessing = data_set_6.review_headline.str.len().mean() + data_set_6.review_body.str.len().mean()

nltk.download("stopwords")

stopwords_set = set(stopwords.words('english'))
remove_list = ['myself', 'what', 'which', 'who', 'whom', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very','can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
for rm_word in remove_list:
    stopwords_set.remove(rm_word)
stopwords_set.add('would')

def remove_stopwords(text):
    result = []
    tokenized = word_tokenize(text)
    for word in tokenized:
        if word not in stopwords_set:
            result.append(word)
    return " ".join(result)

data_set_7 = data_set_6.copy()
data_set_7['review_body'] = data_set_7['review_body'].apply(remove_stopwords)
data_set_7['review_headline'] = data_set_7['review_headline'].apply(remove_stopwords)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    result = []
    tokenized = word_tokenize(text)
    for word in tokenized:
        result.append(lemmatizer.lemmatize(word))
    return " ".join(result)

data_set_8 = data_set_7.copy()
data_set_8['review_body'] = data_set_8['review_body'].apply(lemmatization)
data_set_8['review_headline'] = data_set_8['review_headline'].apply(lemmatization)

data_set_9 = data_set_8.copy()
data_set_9['review'] = data_set_9['review_headline'] + ' ' + data_set_9['review_body']

# concat headline and body, remove extra whitespaces
data_set_10 = data_set_9.copy()
data_set_10 = data_set_10.drop(columns = ['review_headline', 'review_body'])
data_set_10['review'] = data_set_10['review'].replace(r"\s+", ' ', regex=True)
data_set_10['review'] = data_set_10['review'].replace(r"one star", 'onestar', regex=True)
data_set_10['review'] = data_set_10['review'].replace(r"two star", 'twostar', regex=True)
data_set_10['review'] = data_set_10['review'].replace(r"three star", 'threestar', regex=True)
data_set_10['review'] = data_set_10['review'].replace(r"four star", 'fourstar', regex=True)
data_set_10['review'] = data_set_10['review'].replace(r"five star", 'fivestar', regex=True)

data_set_10 = data_set_10.drop(data_set_10[data_set_10['review'].str.len() < 6].index)

avg_len_after_preprocessing = data_set_10.review.str.len().mean()
print(avg_len_before_preprocessing.astype(str) + ', ' + avg_len_after_preprocessing.astype(str))

# label the classes first
def classify(rating):
    if rating == 1 or rating == 2:
        return 1
    elif rating == 3:
        return 2
    else:
        return 3
data_set_10['star_rating'] = data_set_10['star_rating'].apply(classify)


train, test = train_test_split(data_set_10, test_size = 0.2, train_size = 0.8, shuffle = True)

tv = TfidfVectorizer()
tv.fit(train['review'])
X_train_vec = tv.transform(train['review'])
X_test_vec = tv.transform(test['review'])
Y_train = train['star_rating']
Y_test = test['star_rating']

def print_metrics(Y_pred, Y_test):
    precision = precision_score(Y_test, Y_pred, average=None)
    recall = recall_score(Y_test, Y_pred, average=None)
    f1 = f1_score(Y_test, Y_pred, average=None)
    p1, p2, p3 = precision
    r1, r2, r3 = recall
    f_1, f_2, f_3 = f1
    print(p1.astype(str) + ', ' + r1.astype(str) + ', ' + f_1.astype(str))
    print(p2.astype(str) + ', ' + r2.astype(str) + ', ' + f_2.astype(str))
    print(p3.astype(str) + ', ' + r3.astype(str) + ', ' + f_3.astype(str))

    print(f"{np.mean(precision)}, {np.mean(recall)}, {np.mean(f1)}")


clf = Perceptron(max_iter=8000, alpha=0.1, early_stopping=True)

clf.fit(X_train_vec, Y_train)

Y_pred = clf.predict(X_test_vec)

print_metrics(Y_pred, Y_test)


clf = LinearSVC(C=0.1, max_iter=2000)

clf.fit(X_train_vec, Y_train)

Y_pred = clf.predict(X_test_vec)

print_metrics(Y_pred, Y_test)


clf = LogisticRegression(penalty='l2', solver="saga", C=1)

clf.fit(X_train_vec, Y_train)

Y_pred = clf.predict(X_test_vec)

print_metrics(Y_pred, Y_test)



clf = MultinomialNB(alpha=20)

clf.fit(X_train_vec, Y_train)

Y_pred = clf.predict(X_test_vec)

print_metrics(Y_pred, Y_test)











