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


'''
https://www.kaggle.com/code/datafan07/train-your-own-tokenizer
'''


test = pd.read_csv('data/text-detection/test_essays.csv')
sub = pd.read_csv('data/text-detection/sample_submission.csv')
org_train = pd.read_csv('data/text-detection/train_essays.csv')
train = pd.read_csv('data/text-detection/train_v2_drcat_02.csv')
train = train.drop_duplicates(subset='text').reset_index()  # no duplicates


LOWERCASE = False
VOCAB_SIZE = 30522


raw_tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ['[UNK]', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)


dataset = Dataset.from_pandas(test[['text']])


def chunk_iteration(chunk: int = 1_000):
    for i in range(0, len(dataset), chunk):
        yield dataset[i: i+chunk]['text']


raw_tokenizer.train_from_iterator(chunk_iteration(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token='[UNK]', pad_token='[PAD]', cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')


tokenizer_texts_train = []
for t in tqdm(train['text'].tolist()):
    tokenizer_texts_train.append(tokenizer.tokenize(t))


tokenized_texts_test = []
# Tokenize test set with new tokenizer
for t in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(t))


dummy = lambda text: text


vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer='word',
                             tokenizer=dummy, preprocessor=dummy, token_pattern=None, strip_accents='unicode')
vectorizer.fit(tokenized_texts_test)
vocab = vectorizer.vocabulary_


vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer='word', vocabulary=vocab,
                             tokenizer=dummy, preprocessor=dummy, token_pattern=None, strip_accents='unicode')
tf_train = vectorizer.fit_transform(tokenizer_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)
del vectorizer
gc.collect()


y_train = train['label'].values


# CLASSIFIER PIPELINE


bayes_model = MultinomialNB(alpha=0.02)
sgd_model = SGDClassifier(max_iter=8_000, tol=1e-4, loss='modified_huber')
ensemble = VotingClassifier(estimators=[('sgd', sgd_model), ('nb', bayes_model)],
                           weights=[0.7, 0.3], voting='soft', n_jobs=-1)
ensemble.fit(tf_train, y_train)
gc.collect()

final_preds = ensemble.predict_proba(tf_test)[:, 1]
sub['generated'] = final_preds
# sub.to_csv('data/submission.csv', index=False)

















