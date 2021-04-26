import sys
import os
sys.path.append(os.path.join(os.getcwd(),'utility'))

from keras.models import load_model
from utility import functions, globalvars
import librosa
import numpy as np
from train import get_data

emotion_classes=['anger','boredom','disgust','anxiety/fear','happiness','sadness','neutral']


def test(model_path:str):
    U_train, X_train, Y_train, U_test, X_test, Y_test = get_data()
    model=load_model(model_path)
    Y_pred = model.predict([U_test, X_test])
    matrix = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap = 'rocket_r', emotion_classes=['anger','boredom','disgust','anxiety/fear','happiness','sadness','neutral'])
    return matrix
    
    
    
    if __name__ == '__main__':
       model_path='weights_blstm_hyperas_1.h5'
       matrix = test(model_path)
       print(matrix)
       

