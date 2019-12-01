import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from time import perf_counter

def baseline_model():
    # create model
    inputsNr = 10236 #29625 #len(tfidf_vectorizer.get_feature_names())
    outputsNr = 11
    hiddenNodesNr = 100

    model = Sequential()
    model.add(Dense(hiddenNodesNr, input_dim=inputsNr, activation='relu'))
    model.add(Dense(outputsNr, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


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

### citim datele ###

#TRAIN
dir_path = 'data/trainData/'
train_data_path = os.path.join(dir_path, 'trainExamples')

#TEST
dir_path2 = 'data/testData/'
test_data_path = os.path.join(dir_path2, 'testData-public')


texts_train=read_texts(train_data_path)
texts_test = read_texts(test_data_path)

labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))
encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
labels = np_utils.to_categorical(encoded_labels)

texts=texts_train+texts_test
# tfidf_vectorizer = TfidfVectorizer(min_df=1,stop_words='english')
tfidf_vectorizer = TfidfVectorizer(min_df=10)
data = tfidf_vectorizer.fit_transform(texts)
print(data.shape)
#
data_train_indices = np.arange(0, len(texts_train))
data_test_indices=np.arange(len(texts_train), len(texts_train)+len(texts_test))

start_time=perf_counter()
estimator = KerasClassifier(build_fn=baseline_model, epochs=20, verbose=0)
estimator.fit(data[data_train_indices,:], labels[data_train_indices])
stop_time=perf_counter()
print("Timpul de antrenare este: ", stop_time-start_time)
predictii = estimator.predict(data[data_test_indices,:])
print(predictii)

# indici_test=np.arange(2984,4480+1)
indici_test=data_test_indices
predictii = predictii.astype(int)

np.savetxt("data\submisie_kaggle.csv", np.stack((indici_test+1,predictii)).T,fmt="%s",
           delimiter=",",header="Id,Prediction",comments='')



