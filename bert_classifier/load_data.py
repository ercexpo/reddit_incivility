# Load the dataset into a pandas dataframe.
import pandas as pd

def get_data(tsv_file):
    df = pd.read_csv(tsv_file, delimiter=',', names=['comment', 'label'])
    df = df[1:]
    labels = df.label.values
    comments = df.comment.values
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    return comments, labels

