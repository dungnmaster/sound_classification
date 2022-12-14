import pandas as pd
import os
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import logging
from db_writer import DataSink
from datetime import datetime

logging.basicConfig(
    filename='classifier.log',
    filemode='a', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
    level=logging.INFO
)

class NNCLassifier():
    def __init__(self, model_path):
        print('path:',model_path)
        self.model = tf.keras.models.load_model(model_path)
        
        # self.model = tf.keras.models.load_model('model/model_v1.h5')
        # self.model = tf.keras.models.load_model('/home/pi/code/water-sound-classifier/savedModel')
        
        # self.labels = ['flush','shower','silence','sink']
        self.labels = ['flush', 'shower' ,'silence' ,'sink' ,'speech']

        self.sample_duration = 4
        self.db = DataSink()
        self.write_records = []
        print('done')

    def classify(self, filepath):
        print('CLASSIFYING...')
        audio, sample_rate = librosa.load(filepath, res_type='kaiser_fast') 
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
        predicted_label=self.model.predict(mfccs_scaled_features)
        
        # print("##",predicted_label)
        classes_x=np.argmax(predicted_label,axis=1)
        prediction_class = self.labels[int(classes_x)]

        starttime = filepath.split('/')[-1].split('.')[0]
        if not starttime.isnumeric():
            logging.error("segment file name should be start timestamp...offset calculation will FAIL")
        starttime = int(starttime) if starttime.isnumeric() else 0
        endtime = starttime+self.sample_duration

        s_timestamp = datetime.fromtimestamp(starttime, tz=None) 
        e_timestamp = datetime.fromtimestamp(endtime, tz=None)
        duration = self.sample_duration
        label = prediction_class
        
        self.write_records.append([s_timestamp, e_timestamp, label, duration])
        
        # if len(self.write_records) == 5:
        #     if self.write_records[0][2] == self.write_records[1][2] == self.write_records[3][2] == self.write_records[4][2] and self.write_records[0][2] != self.write_records[2][2]:
        #         print('modifying label: '+self.write_records[2][2]+' to '+self.write_records[1][2])
        #         self.write_records[2][2] = self.write_records[0][2]
        #     # print('curr', self.write_records[0])
        #     logging.info(f'starttime: {self.write_records[2][0]} endtime: {self.write_records[2][1]} label: {self.write_records[2][2]} duration: {self.write_records[2][3]}')
        #     self.db.insert([self.write_records[0]])
        #     self.write_records.pop(0)


        if len(self.write_records) == 3:
            if self.write_records[0][2] == self.write_records[2][2]  and self.write_records[0][2] != self.write_records[1][2]:
                print('modifying label: '+self.write_records[1][2]+' to '+self.write_records[0][2])
                self.write_records[1][2] = self.write_records[0][2]
            # print('curr', self.write_records[0])
            logging.info(f'starttime: {self.write_records[1][0]} endtime: {self.write_records[1][1]} label: {self.write_records[1][2]} duration: {self.write_records[1][3]}')
            self.db.insert([self.write_records[0]])
            self.write_records.pop(0)




