import pandas as pd
import torch
from transformers import BertTokenizer
import load_data
import sys, time, datetime, random, csv
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model = BertForSequenceClassification.from_pretrained('/app/incivility/models/bert_5000_03-06-20')
#config = BertConfig.from_json_file('../models/bert_classifier_2epoch_256size/config.json')
tokenizer = BertTokenizer.from_pretrained('/app/incivility/models/bert_5000_03-06-20')

model.cuda()

#load comments and labels from the input tsv
comments, labels = load_data.get_data(sys.argv[1])

#encode inputs using BERT tokenizer
input_ids = []

for comment in comments:
    encoded_comment = tokenizer.encode(comment, add_special_tokens = True, max_length=256,pad_to_max_length=True)
    input_ids.append(encoded_comment)

#define attention masks: if 0 it's a PAD, set to 0; else set to 1
attention_masks = []

for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

labels = labels.astype(np.int)

batch_size = 16

# Create the DataLoader for our training set.
# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables
predictions , true_labels = [], []

# Predict
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch

  # Telling the model not to compute or store gradients, saving memory and
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()

  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')

# Combine the predictions for each batch into a single list of 0s and 1s.
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

with open('../data/predictions/incivility_predictions_5000_03-06-20.tsv', mode='w') as csv_file:
  csv_writer = csv.writer(csv_file, delimiter = '\t', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
  for comment, prediction in zip(comments, flat_predictions):
    csv_writer.writerow([comment, prediction])
