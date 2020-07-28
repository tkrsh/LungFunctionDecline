import pandas as pd 
import tensorflow_io as tfio 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn import Kmea
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from   tensorflow.keras import Sequential
from sklearn.preprocessing import LabelEncoder

def generate_train_df(df):
    train_data=df.drop("Patient",axis=1)
    train_data=train_data.drop("FVC",axis=1)
    train_target=df.pop("FVC")
    train_data["Sex"]=le.fit_transform(df["Sex"])
    train_data["SmokingStatus"]=le.fit_transform(df["SmokingStatus"])
    dataset = tf.data.Dataset.from_tensor_slices((train_data.values, train_target.values))
    train_dataset=dataset.batch(356)
    return train_dataset 

C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

def kloss(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 1]
    fvc_pred = y_pred[:, 0]
    
    sigma_clip = sigma + C1
    #sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    #delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)

def kmae(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    spread = tf.abs( (y_true[:, 0] -  y_pred[:, 0])  / (y_pred[:, 0] + 1.) )
    #spread = tf.abs( (y_true[:, 0] -  y_pred[:, 0])  / y_true[:, 0] )
    return K.mean(spread)

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * kloss(y_true, y_pred) + (1 - _lambda)*kmae(y_true, y_pred)
    return loss

def get_compiled_model():
    model = Sequential([
    L.Dense(1024, activation='relu'),
    L.Dense(1024, activation='relu'),
    L.Dense(1024, activation='relu'),
    L.Dense(256, activation='relu'),
    L.Dense(256, activation='relu'),
    L.Dense(256, activation='relu'),
    L.Dense(126, activation='relu'),
    L.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2)])
    model.compile(loss=mloss(0.5), optimizer="adam", metrics=[kloss])
    return model

def run_model():
    model = get_compiled_model()
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/home/tkrsh/osic-git/", profile_batch=5)
    history=model.fit(generate_train_df(df),epochs=2000)

if __name__== "__main__":

    K.clear_session()
    le=LabelEncoder()
    df=pd.read_csv("/home/tkrsh/osic-git/csvs/train.csv")
    run_model()