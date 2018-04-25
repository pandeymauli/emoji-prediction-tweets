from sklearn.metrics import f1_score
import json
import numpy as np
import random

f_en = open('Dev_Data/us_trial.labels')
f_es = open('Dev_Data/es_trial.labels')

y_en = []
for line in f_en:
    y_en.append(int(line.strip()))
f_en.close()

y_es = []
for line in f_es:
    y_es.append(int(line.strip()))
f_es.close()

random_int = random.randint(0,19)
print(random_int)
y_en_random = [random_int for i in range(len(y_en))]
y_es_random = [random_int for i in range(len(y_es))]

score_en = f1_score(y_en, y_en_random, average='weighted', labels=np.unique(y_en_random) )
score_es = f1_score(y_es, y_es_random, average='weighted', labels=np.unique(y_es_random) )

print("English Baseline",score_en)
print("Spanish Baseline",score_es)

cache_score = open('F-scores.txt','r')
txt = cache_score.read()
cache_score.close()
if txt:
    data = json.loads(txt.strip())
else:
    data = {}
data['FScore_Baseline_English'] = score_en
data['FScore_Baseline_Spanish'] = score_es
print("DATA",data)
cache_score = open('F-scores.txt','w')
cache_score.write(json.dumps(data))
