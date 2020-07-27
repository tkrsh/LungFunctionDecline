import tensorflow as tf
import pandas as pd 
import tensorflow_io as tfio 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import os
main_dir=("/home/tkrsh/osic-main/")

files=[]

for dirname, _, filenames in os.walk(main_dir):
    for filename in (filenames):
        files.append(os.path.join((dirname), filename))

files=[x for x in files if '.csv' not in x]

train_images= [str(x) for x in files if 'train'  in x]
test_images = [str(x) for x in files if 'test'   in x]

def decode_image(image_path):
    image_bytes = tf.io.read_file(image_path)
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16)
    image=np.squeeze(image.numpy())
    return image

def show_scan(image):
    img = decode_image(image)
    patient_name=str(image).split('/')[1]
    fig, ax = plt.subplots()
    im=ax.imshow(img,cmap='Greys')
    plt.axis('off')
    plt.title("Baseline CT Scan of Patient {}".format(patient_name))
    fig.set_size_inches(9,9)
    plt.show()

train=pd.read_csv(main_dir+"train.csv")

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df['Sex']=le.fit_transform(df['Sex'])