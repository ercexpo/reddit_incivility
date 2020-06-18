import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import csv
import nltk
import spacy
from imblearn.over_sampling import ADASYN

stemmer = nltk.stem.snowball.SnowballStemmer(language='english')
sp = spacy.load('en_core_web_sm')

def stem_each(input_str):
    toks = nltk.word_tokenize(input_str)
    stems = []
    for tok in toks:
        stems.append(stemmer.stem(tok))
    out_str = ' '.join(stems)
    return out_str

df = pd.read_csv('../data/split/incivility_coded_0302.train.csv', delimiter=',',header=None)

#X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[0],df[1])
X_train_raw = df[0].str.lower()
y_train = df[1]

X_train_stemmed = X_train_raw.apply(stem_each)
print(X_train_stemmed)

vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
X_train = vectorizer.fit_transform(X_train_stemmed)
oversample = ADASYN()
X_train, y_train = oversample.fit_resample(X=X_train, y=y_train)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

df2 = df = pd.read_csv('../data/split/incivility_coded_0302.test.csv', delimiter=',',header=None)

X_test_raw = df2[0]
X_test_lower = df2[0].str.lower()
X_test_stemmed = X_test_lower.apply(stem_each)
y_test = df2[1]

X_test = vectorizer.transform(X_test_stemmed)
predictions = classifier.predict(X_test)

csv_writer = csv.writer(open('predictions_os.tsv', mode='w'), delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

for comment, prediction in zip(X_test_raw, predictions):
    csv_writer.writerow([comment, prediction])
