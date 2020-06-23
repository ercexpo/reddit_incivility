# reddit_incivility
Classification of incivility in Reddit posts

# Replication scripts for prediction of incivility in Reddit posts.

We first train a classification model using BERT and its close relative DistilBERT. While we achieve good accuracy and F1 with these models, they are computationally expensive given the large amount of Reddit data we wish to classify. 

Therefore we also train a basic logistic regression model using TF-IDF feature. However, when trained on our annotated training data alone, the performance of the logistic regression model is inadequate.

To generate additional synthethic labeled data for training our logistic regression model, we utilize an uptraining procedure. We use our best performing DistilBERT model to classify 5 million Reddit posts for incivility. We then combine this synthetic data with our annotated data and train a logistic regression classifier on this new training data. Using this approach we are able to achieve the following performance.

```
Accuracy	0.958
Precision	0.852
Recall	        0.721
F1	        0.781
```

To replicate our setup, we recommend the following steps:

1) Split annotated training data into train/test sets using the file in `scripts/train_test_split.py`
2) Train DistilBERT model using the train set by running `bert_classifier/classify_distilbert` with input csv file and desired output model name as arguments.
3) Use the script `run_model_distilbert_large.py` with bz2 file of Reddit data as an argument. You will need to modify the script to point to the correct model file from the previous step and to specify the output file.
4) Combine desired amount of output from step 3 with the test dataset from step 1. Train a log regression model using this data - `log_regression/log_regression_uptrain.py`. Please modify the file to indicate the correct input file and test file. Output will be a trained model and a file containing predictions of the test data input.

Trained Models can be found here: [BERT](https://drive.google.com/file/d/1EAS2kEKp4bDO0s657P3hIGh5jZ1qZM2T/view?usp=sharing), [DistilBERT](https://drive.google.com/file/d/1fTNHZigPX_TzOHZCisjJgh88BDR_Lhrd/view?usp=sharing)

Please contact ssdavidson@ucdavis.edu with any questions regarding this implementation.


