# from sklearn.metrics import roc_auc_score
#
# # Example true labels and predicted scores
# y_true = [0, 1, 1, 0, 1, 0]
# y_scores = [0.2, 0.8, 0.6, 0.3, 0.7, 0.4]
#
# # Calculate ROC AUC
# auc = roc_auc_score(y_true, y_scores)
#
# print(f'ROC AUC: {auc}')


import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier


test = pd.read_csv('data/text-detection/test_essays.csv')
sub = pd.read_csv('data/text-detection/sample_submission.csv')
org_train = pd.read_csv('data/text-detection/train_essays.csv')
train = pd.read_csv('data/text-detection/train_v2_drcat_02.csv')












