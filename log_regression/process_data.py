from google.cloud import bigquery
from google.oauth2 import service_account
import sys, multiprocessing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, cross_val_score
import time, sys, math, pickle

credentials = service_account.Credentials.from_service_account_file(
'../api_keys/incivility-reddit-b1d51017d1d6.json')

classifier = pickle.load(open('../incivility_project/log_regression/models/model.pkl', mode='rb'))
vectorizer = pickle.load(open('../incivility_project/log_regression/models/vectorizer.pkl', mode='rb'))

project_id = 'incivility-reddit'
dataset = 'reddit_comments'

year = sys.argv[1]

data_source = "`fh-bigquery.reddit_comments." + year + "`"

def process_dataframe(source, num_per_grab, i):
    client = bigquery.Client(credentials=credentials, project=project_id)
    data = client.query("SELECT * FROM " + source + " LIMIT {} OFFSET {}".format(num_per_grab, i), location='US').to_dataframe()

    num_chunks = math.ceil(len(data)/10000)
    for chunk in np.array_split(data, num_chunks):

        print(chunk)

        print('Predicting labels for 10000 sentences...')

        t0 = time.time()

        subreddit_list = chunk["subreddit"]
        subreddit_id_list = chunk["subreddit_id"]
        comments = chunk.body
        comments_lower = comments.str.lower()

        X_test = vectorizer.transform(comments_lower)
        predictions = classifier.predict(X_test)

        out_data = {'comment':comments, 'subreddit':subreddit_list, 'subreddit_id':subreddit_id_list, 'prediction': predictions}
        out_frame = pd.DataFrame(out_data)

        target = "reddit_predictions." + year

        out_frame.to_gbq(target, 'incivility-reddit', chunksize=None, if_exists='append')

        print('    DONE.')
        print("  Inference took: {:}".format(time.time() - t0))


if __name__ == "__main__":
    with multiprocessing.Manager() as manager:

        temp_client = bigquery.Client(credentials=credentials, project=project_id)

        length = temp_client.query("SELECT COUNT(*) FROM " + data_source, location='US').to_dataframe().f0_[0]

        num_per_grab = math.ceil(length/15)

        split_frames = []

        starttime = time.time()
        processes = []

#        for i in range(0, length, num_per_grab):
        #testing
        for i in range(1):
#            frame = client.query("SELECT * FROM " + source + " LIMIT {} OFFSET {}".format(num_per_grab, i), location='US').to_dataframe()
            p = multiprocessing.Process(target=process_dataframe, args = (data_source, num_per_grab, i))
            processes.append(p)
            p.start()
#            split_frames.append(frame)

        # starttime = time.time()
        # processes = []
        # for frame in split_frames:
        #     p = multiprocessing.Process(target=process_dataframe, args = (frame))
        #     processes.append(p)
        #     p.start()

        for process in processes:
            process.join()
