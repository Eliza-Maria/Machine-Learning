import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from time import perf_counter


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


#TRAIN
dir_path = 'data/trainData/'
train_data_path = os.path.join(dir_path, 'trainExamples')

#TEST
dir_path2 = 'data/testData/'
test_data_path = os.path.join(dir_path2, 'testData-public')


texts_train=read_texts(train_data_path)
texts_test = read_texts(test_data_path)
labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))

texts=texts_train+texts_test
# tfidf_vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
tfidf_vectorizer = TfidfVectorizer()
data = tfidf_vectorizer.fit_transform(texts)

data_train_indices = np.arange(0, len(texts_train))
data_test_indices=np.arange(len(texts_train), len(texts_train)+len(texts_test))

print(data_test_indices)

start_time=perf_counter()
clf = LinearSVC(C=1, loss='squared_hinge', max_iter=15000)
clf.fit(data[data_train_indices, :], labels[data_train_indices])
stop_time=perf_counter()
print("Timpul de antrenare este: ", stop_time-start_time)
predictions=clf.predict(data[data_test_indices, :])


predictions = predictions.astype(int)

np.savetxt("data\submisie_kaggle.csv", np.stack((data_test_indices+1,predictions)).T,fmt="%s",
           delimiter=",",header="Id,Prediction",comments='')

