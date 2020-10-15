import pandas as pd 
import tensorflow_io as tfio 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from   tensorflow.keras import Sequential
main_dir=("/home/tkrsh/osic-main/")
 
files=[]

for dirname, _, filenames in os.walk(main_dir):
    for filename in (filenames):
        files.append(os.path.join((dirname), filename))

files=[x for x in files if '.csv' not in x]

train_images= [str(x) for x in files if 'train'  in x]
test_images = [str(x) for x in files if 'test'   in x]

train=pd.read_csv(main_dir+"train.csv")

df=train.copy()

df=pd.concat([df,pd.get_dummies(df['Sex'])],axis=1).drop(['Sex'],axis=1)
df=pd.concat([df,pd.get_dummies(df['SmokingStatus'])],axis=1).drop(['SmokingStatus'],axis=1)
df['dFVC'] =df["FVC"]-df["FVC"].shift(1)
df['d%'] = df["Percent"]-df["Percent"].shift(1)
df['Gap'] = df["Weeks"]-df["Weeks"].shift(1)
df.fillna(0,inplace=True)
Means=KMeans(n_clusters=3).fit((df["Age"].values).reshape(-1,1))
df["Age_Cat"]=Means.labels_
df=pd.concat([df,pd.get_dummies(df['Age_Cat'],prefix="Age_Cat")],axis=1).drop(['Age_Cat'],axis=1)
train_df=df.drop("Patient",axis=1)
df["Gap"]= [int(x) for x in df["Gap"]]
x_train=df.drop("Patient",axis=1)
x_train=x_train.drop("Age",axis=1)
x_train=x_train.drop("FVC",axis=1)
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)


def kloss(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 1]
    fvc_pred = y_pred[:, 0]
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)

def kmae(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    spread = tf.abs( (y_true[:, 0] -  y_pred[:, 0])  / y_true[:, 0] )
    #spred = tf.square(y_true, y_pred[:, 0])
    return K.mean(spread)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * kloss(y_true, y_pred) + (1 - _lambda)*kmae(y_true, y_pred)
    return loss




def get_compiled_model():
    model = tf.keras.Sequential([
    tf.keras.layers.LSTM(10,return_sequences=False),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
            ])


    model.compile(loss=mloss(0.5), optimizer="adam", metrics=[kloss])
    return model
tf.keras.backend.clear_session()

y_train=df.pop("FVC")
 
dataset = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
train_dataset=dataset.batch(9)
model = get_compiled_model()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/home/tkrsh/osic-git/", profile_batch=5)

model.fit(train_dataset,epochs=20000,callbacks=[tensorboard_callback])

model.save_weights("Model_Over_50k")
model.save("OVERFITT")

