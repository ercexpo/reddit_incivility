import pandas as pd
from numpy.random import RandomState
import sys

df = pd.read_csv(sys.argv[1])

train_out = sys.argv[1][:-3] + "train.csv"
test_out = sys.argv[1][:-3] + "test.csv"

rng = RandomState()

train = df.sample(frac=0.8, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

#print(test)

#train = train[['comment','correction']]
#test = test[['comment','correction']]

#print(test)

train.to_csv(train_out, index=False)
test.to_csv(test_out, index=False)
