import numpy as np
import os
import random
import math
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix

def get_accuracy(y, p):
    return 100 * (y == p).astype('int').mean()


def files_in_folder(mypath):
    files = []
    for f in os.listdir(mypath):
        if os.path.isfile(os.path.join(mypath, f)):
            files.append(os.path.join(mypath, f))
    return sorted(files)


def extract_file_without_extention(cale_catre_fisier):
    file_name= os.path.basename(cale_catre_fisier)
    file_name_without_extension = file_name.replace('.txt', '')
    return file_name_without_extension


def read_texts(cale):
    texts = []
    for file in files_in_folder(cale):
        id_file = extract_file_without_extention(file)
        with open(file, 'r', encoding='utf-8') as fin:
            text = file.read()
        texts.append(text)
    return texts

dir_path = 'data/trainData/'

#TRAIN
train_data_path = os.path.join(dir_path, 'trainExamples')
texts_train=read_texts(train_data_path)

labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))
labels=labels.astype(int)

tfidf_vectorizer = TfidfVectorizer(min_df=1)
data_train=tfidf_vectorizer.fit_transform(texts_train)

indici_date_train = np.arange(0, len(texts_train))
random.shuffle(indici_date_train)

no_partitions = 10  # !!!

no_test_examples = math.floor(len(texts_train) / no_partitions)

for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    accuracy = 0
    for idx_partition in range(0, 10):

        min_idx_test = idx_partition * no_test_examples

        if idx_partition != 9:
            max_id_test = (idx_partition + 1) * no_test_examples - 1
        else:
            max_id_test = len(texts_train)

        #print(min_idx_test, max_id_test)

        train_indices = []
        test_indices = []
        for i, idx in enumerate(indici_date_train):
            if i >= min_idx_test and i < max_id_test:
                test_indices.append(idx)
            else:
                train_indices.append(idx)

        clf = LinearSVC(C=C, loss='squared_hinge', max_iter=15000)#random_state=0, tol=1e-5)
        clf.fit(data_train[train_indices, :], labels[train_indices])
        predictions=clf.predict(data_train[test_indices, :])
        print("Accuracy at step ", idx_partition ,": ", get_accuracy(predictions, labels[test_indices]))
        accuracy += get_accuracy(predictions, labels[test_indices])

        matrice_confuzie=confusion_matrix(labels[test_indices], predictions)
        print(matrice_confuzie)

    print("Acuratetea medie cu C =", C, ": ", accuracy / no_partitions)


