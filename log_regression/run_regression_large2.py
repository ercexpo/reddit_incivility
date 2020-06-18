import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score
import csv, pickle, time, sys
#import spacy
#from imblearn.over_sampling import ADASYN
#from imblearn.over_sampling import SMOTE
from CustomIterableDataset import CustomIterableDataset
from torch.utils.data import DataLoader

classifier = pickle.load(open('models/model.pkl', mode='rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', mode='rb'))

#vectorizer = TfidfVectorizer(vocabulary=pickle.load(open('models/tfidf_vocab.pkl', mode='rb')))

classifier.n_jobs = 16
vectorizer.n_jobs = 16

batch_size = 20000

dataset = CustomIterableDataset(sys.argv[1])
dataloader = DataLoader(dataset, batch_size = batch_size)

csv_writer = csv.writer(open('predictions/' + sys.argv[2], mode='w'), delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)

for data in dataloader:

    print('Predicting labels for 5000 test sentences...')
#    print()
#    print(data)
    t0 = time.time()

#encode inputs using BERT tokenizer
    labels = data[3]
    subreddit_list = data[1]
    subreddit_id_list = data[2]
    comments = data[0]

#    for comment in data:
        #print(comment)
        #print("BREAK")

#        if comment[1] != '':
#            text = comment[0]
#            label = comment[3]
#            subreddit = comment[1]
#            subreddit_id = comment[2]

#            labels.append(label)
#            subreddit_list.append(subreddit)
#            subreddit_id_list.append(subreddit_id)
#            comments.append(text)
#        else:
#            continue

    #print(comments)

    data = {'comment':comments, 'subreddit_list':subreddit_list, 'subreddit_id_list':subreddit_id_list}
#    data = pd.DataFrame(zip(comments, subreddit_list, subreddit_id_list))
    data = pd.DataFrame(data)
    data.dropna(axis=0, inplace=True)

#    print(data)

    comments = data['comment']
    #print(comments)
    comments_lower = comments.str.lower()
    subreddit_list = data['subreddit_list']
    subreddit_id_list = data['subreddit_id_list']

    X_test = vectorizer.transform(comments_lower)
    predictions = classifier.predict(X_test)

    for comment, prediction, subreddit, subreddit_id in zip(comments, predictions, subreddit_list, subreddit_id_list):
        try:
            csv_writer.writerow([comment, prediction, subreddit, subreddit_id])
        except:
            continue

    print('    DONE.')
    print("  Inference took: {:}".format(time.time() - t0))


#mapping function to process input
def line_mapper(self, line):

    #Splits the line into text and label and applies preprocessing to the text

    try:
        data = yaml.load(line, Loader=Loader)
    except:
        return ('','','',0)

    if data['author'] == 'AutoModerator':
        return ('', '', '', 0)

    text = data['body']
    subreddit = data['subreddit']
    subreddit_id = data['subreddit_id']
    text = self.preprocess(text)
    label = 0

#        print((text, subreddit, subreddit_id, label))

    return (text, subreddit, subreddit_id, label)
