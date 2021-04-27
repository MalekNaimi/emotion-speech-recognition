import sys
import os
sys.path.append(os.path.join(os.getcwd(),'utility'))

from keras.models import load_model
from utility import functions, globalvars
import librosa
import numpy as np
import seaborn as sns
from train import get_data
from sklearn.metrics import confusion_matrix

emotion_classes=['anger','boredom','disgust','anxiety/fear','happiness','sadness','neutral']
U_train, X_train, Y_train, U_test, X_test, Y_test = get_data()
model_path='weights_blstm_hyperas_1.h5'
model=load_model(model_path)
Y_pred = model.predict([U_test, X_test])
matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
ax = sns.heatmap(matrix, annot=True, fmt="d", cmap = 'rocket_r', xticklabels = ['anger','boredom','disgust','anxiety/fear','happiness','sadness','neutral'], yticklabels = ['anger','boredom','disgust','anxiety/fear','happiness','sadness','neutral'])
print(matrix)

