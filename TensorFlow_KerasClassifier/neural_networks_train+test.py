import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def baseline_model():
    # create model
    inputsNr = 8470 #29625 #len(tfidf_vectorizer.get_feature_names())
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

texts_train=read_texts(train_data_path)
labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))

tfidf_vectorizer = TfidfVectorizer(min_df=10)
data_train=tfidf_vectorizer.fit_transform(texts_train)
print(data_train.shape)

print(data_train[0,:].shape)

encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
labels = np_utils.to_categorical(encoded_labels)

estimator = KerasClassifier(build_fn=baseline_model, epochs=20, verbose=10)
kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, data_train, labels, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
