import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score
import csv, pickle
import nltk
#import spacy
#from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE

#stemmer = nltk.stem.snowball.SnowballStemmer(language='english')
#sp = spacy.load('en_core_web_sm')

# def stem_each(input_str):
#     toks = nltk.word_tokenize(input_str)
#     stems = []
#     for tok in toks:
#         stems.append(stemmer.stem(tok))
#     out_str = ' '.join(stems)
#     return out_str

df = pd.read_csv('../results/incivility_predictions_large_test.tsv', delimiter='\t',header=None, nrows=500000)
df.dropna(axis=0, inplace=True)
print(df)

#X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[0],df[1])
X_train_raw = df[0].str.lower()
print(X_train_raw)
y_train = df[3]

#X_train_stemmed = X_train_raw.apply(stem_each)
#print(X_train_stemmed)

vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
#X_train = vectorizer.fit_transform(X_train_stemmed)
X_train = vectorizer.fit_transform(X_train_raw)
pickle.dump(vectorizer.vocabulary_, open('tfidf_vocab.pkl', mode="wb"))
pickle.dump(vectorizer, open('vectorizer.pkl', mode="wb"))
print("done with vectorizer")
#oversample = ADASYN()
oversample = SMOTE(n_jobs=12)
X_train, y_train = oversample.fit_sample(X=X_train, y=y_train)
print(X_train)
print(y_train)
classifier = LogisticRegression(n_jobs=12, solver='sag')
classifier.fit(X_train, y_train)

print("Model fitting complete")

df2 = pd.read_csv('../../incivility/data/split/incivility_coded_0302.test.csv', delimiter=',',header=None, skiprows=1)

X_test_raw = df2[0]
X_test_lower = df2[0].str.lower()
#X_test_stemmed = X_test_lower.apply(stem_each)
y_test = df2[1]

#X_test = vectorizer.transform(X_test_stemmed)
X_test = vectorizer.transform(X_test_lower)
predictions = classifier.predict(X_test)

csv_writer = csv.writer(open('predictions_os_uptrain.tsv', mode='w'), delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

for comment, label, prediction in zip(X_test_raw, y_test, predictions):
    csv_writer.writerow([comment, label, prediction])

pickle_file = open("model.pkl", mode="wb")
pickle.dump(classifier, pickle_file)
