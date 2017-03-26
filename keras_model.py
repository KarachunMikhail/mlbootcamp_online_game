# -*- coding: utf-8 -*-

from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adamax
from sklearn.preprocessing import  StandardScaler


class KerasModel(object):
    def __init__(self,variables, epoche):
        self.scaler = StandardScaler()
        self.epoche = epoche
        self.model = Sequential()
        model = self.model

        model.add(Dense(60,input_dim=len(variables)))
        model.add(LeakyReLU())
        model.add(Dropout(0.09))

        model.add(Dense(30,input_dim=len(variables)))
        model.add(LeakyReLU())
        model.add(Dropout(0.06))


        model.add(Dense(1, activation='sigmoid'))
        opt = Adamax(lr=0.1)

        model.compile(loss='binary_crossentropy', optimizer=opt)



    def fit_process(self,X, Y):
        self.scaler.fit(X)

    def process(self,X):
        return self.scaler.transform(X)


    def fit(self, X,y, sample_weight=None, callbacks=[]):
        process_X = self.process(X)
        process_y = y
        self.model.fit(process_X, process_y, batch_size=512,
                       nb_epoch=self.epoche, verbose=0,
                       sample_weight=sample_weight,
                       callbacks=callbacks,
                       validation_split=0.02,
                       shuffle=True)

    def predict_proba(self, X):
        process_x = self.process(X)
        result  = self.model.predict(process_x)
        return result
