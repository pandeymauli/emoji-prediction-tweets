import csv
import re
import itertools
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import math
from sklearn.metrics import f1_score
import sys
import numpy as np
import json

imp_files = open('files.txt').readlines()

if sys.argv[1] == 'en':
    # English Tweets Related Files
    training_data_file = imp_files[0].strip()
    dev_data_txt_file = imp_files[1].strip()
    dev_data_labels_file = imp_files[2].strip()
    test_data_file = imp_files[3].strip()
    test_predict_write = imp_files[4].strip()

if sys.argv[1] == 'es':
    # Spanish Tweets Related Files
    training_data_file = imp_files[5].strip()
    dev_data_txt_file = imp_files[6].strip()
    dev_data_labels_file = imp_files[7].strip()
    test_data_file = imp_files[8].strip()
    test_predict_write = imp_files[9].strip()


def train(filename, smoothing_alpha = 1):
    f_tweets = open(filename, 'r')
    f_tweets.readline()

    print("Reading from {}".format(filename))
    print("Training the classifier... firing on all cylinders!")
    vocabDict = {}

    totalLabels = 19
    if sys.argv[1] == 'en':
        totalLabels += 1

    count_state_words = [ 0 for i in range(totalLabels) ]
    count_state_tweets =  [ 0 for i in range(totalLabels) ]

    for line in f_tweets:
        line = line.strip().split('\t')
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(line[0])
        label = int(line[1].strip())

        count_state_tweets[label] += 1

        for t in tokens:
            if t not in vocabDict:
                list_t = [ 0 for i in range(totalLabels) ]
                vocabDict[t] = list_t
            vocabDict[t][label] = vocabDict[t][label] + 1
            count_state_words[label] += 1

    len_V = len(vocabDict)
    for key in vocabDict:
        for l in range(totalLabels):
            vocabDict[key][l] = (vocabDict[key][l]+smoothing_alpha)/(len_V + count_state_words[l])

    prob_states = [ 0 for i in range(totalLabels) ]
    total_counts = sum(count_state_tweets)
    for l in range(totalLabels):
        prob_states[l] = count_state_tweets[l] / total_counts
    #print(count_state_words, count_state_tweets, len(vocabDict), prob_states)
    f_tweets.close()
    return vocabDict, prob_states


def classify(filenames, V, P):
    fout_tweets = open(filenames[0], 'r')
    fout_classified = open(filenames[1],'w')
    y_dev_pred = []
    for line in fout_tweets:
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(line.strip())
        prob_all_labels = [ math.log(P[i]) for i in range(len(P)) ]
        for t in tokens:
            if t not in V:
                continue
            for l in range(len(P)):
                if V[t][l]:
                    prob_all_labels[l] += math.log(V[t][l])
        max_prob = max(prob_all_labels)
        max_prob_index = prob_all_labels.index(max_prob)
        y_dev_pred.append(max_prob_index)
        fout_classified.write(str(max_prob_index)+'\n')

    fout_tweets.close()
    fout_classified.close()
    return y_dev_pred

smoothing_alpha = 1
print("\n ==================================================== \n")
V, P = train(training_data_file,smoothing_alpha)
print("\n ==================================================== \n")
print("Using dev dataset to make prediction... be patient now!")
y_dev_pred = classify([dev_data_txt_file,'Dev_Data/'+sys.argv[1]+'.text'], V, P)

y_dev = []
f = open(dev_data_labels_file)
for line in f:
    y_dev.append(int(line.strip()))
f.close()

labels = [i for i in range(len(P))]
print("\n ==================================================== \n")
print("Hold your breath for F-Score on dev dataset")
score = f1_score(y_dev, y_dev_pred, average='weighted', labels=np.unique(y_dev_pred) )
print(score)

cache_score = open('F-scores.txt','r')
txt = cache_score.read()
cache_score.close()
if txt:
    data = json.loads(txt.strip())
else:
    data = {}
print("DATA",data)
if sys.argv[1] == 'en':
    key = 'FScore_NaiveBayes_English_alpha_' + str(smoothing_alpha)
    data[key] = score
else:
    key = 'FScore_NaiveBayes_Spanish_alpha_' + str(smoothing_alpha)
    data[key] = score
cache_score = open('F-scores.txt','w')
cache_score.write(json.dumps(data))
cache_score.close()

print("\n ==================================================== \n")
print("Finally... Prediction on test set!")
classify([test_data_file,test_predict_write], V, P)
print("\n ==================================================== \n")
print("Check Test_Data directory for fruit of your hard work!")
