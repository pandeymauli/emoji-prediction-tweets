from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import random
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
from skmultilearn.dataset import load_dataset as Dataset
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import TweetTokenizer
from scipy.sparse import csr_matrix
import scipy.sparse
import numpy as np
from collections import Counter
import sys
import csv
import json
from sklearn.metrics import f1_score

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

print("\n ==================================================== \n")
data = []
data_labels = []
tknzr = TweetTokenizer()

def convert_to_matrix(filename):
    f = open(filename, 'r')
    f.readline()
    print("Reading from {}".format(filename))
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    Y_vector = []

    for row in f:
        #print(row)
        r = row.strip().split('\t')
        d = tknzr.tokenize(r[0])
        Y_vector.append(int(r[1]))
        for term in d:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    X_matrix = csr_matrix((data, indices, indptr), dtype=int)
    Y_vector = np.array(Y_vector)
    f.close()
    return X_matrix, Y_vector, vocabulary

X, y, vocab = convert_to_matrix(training_data_file)

print("\n ==================================================== \n")
print("Training the classifier... firing on all cylinders!")
log_model = LogisticRegression(random_state=0,solver='saga',multi_class="multinomial")
log_model = log_model.fit(X=X, y=y)

orderedVocab = sorted(list(vocab.items()), key = lambda x: x[1])

y_dev_pred = []
f_dev = open(dev_data_txt_file)
print("\n ==================================================== \n")
print("Using dev dataset to make prediction... be patient now!")
print("\n ==================================================== \n")
num = 0
for line in f_dev:
    num += 1
    tokens = tknzr.tokenize(line.strip())
    freqs = Counter(tokens)
    p = []
    for i in orderedVocab:
        if i[0] in tokens:
            count = freqs[i[0]]
            p.append(count)
        else:
            p.append(0)
    pcr = csr_matrix((p),dtype=int)
    y_hat = log_model.predict(pcr)
    print("Predicting for tweet in dev", num)
    y_dev_pred.append(y_hat[0])
f_dev.close()

y_dev = []
f = open(dev_data_labels_file)
for line in f:
    y_dev.append(int(line.strip()))
f.close()

labels = [i for i in range(20)]
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
if sys.argv[1] == 'en':
    data['FScore_Regression_English'] = score
else:
    data['FScore_Regression_Spanish'] = score
cache_score = open('F-scores.txt','w')
cache_score.write(json.dumps(data))
cache_score.close()

print("\n ==================================================== \n")
print("Finally... Prediction on test set!")

f_test_write = open(test_predict_write,'w')
with open(test_data_file) as f_test_read:
        num = 0
        for line in f_test_read:
            num += 1
            tokens = tknzr.tokenize(line.strip())
            freqs = Counter(tokens)
            p = []
            for i in orderedVocab:
                if i[0] in tokens:
                    count = freqs[i[0]]
                    p.append(count)
                else:
                    p.append(0)
            pcr = csr_matrix((p),dtype=int)
            y_hat = log_model.predict(pcr)
            print("Predicting for tweet in test ", num, " : ", y_hat)
            row = line.strip() + '\t' + str(y_hat[0]) + '\n'
            f_test_write.write(row)
        f_test_read.close()
f_test_write.close()

print("\n ==================================================== \n")
print("Check Test_Data directory for fruit of your hard work!")
