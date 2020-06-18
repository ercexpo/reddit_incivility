from torch.utils.data import IterableDataset
import bz2, yaml
from yaml import Loader

class CustomIterableDataset(IterableDataset):

    def __init__(self, filename):

        #Store the filename in object's memory
        self.filename = filename

        #And that's it, we no longer need to store the contents in the memory

    def preprocess(self, text):

        ### Do something with text here
        text_pp = text.lower().strip()
        ###

        return text_pp

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


    def __iter__(self):

        #Create an iterator
        file_itr = bz2.BZ2File(self.filename)

        #Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr
