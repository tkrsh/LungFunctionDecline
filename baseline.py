import tensorflow as tf
import pandas as pd 
import tensorfow_io as tfio 
import sklearn
import os
main_dir=("/home/tkrsh/osic-main/")

files=[]

for dirname, _, filenames in os.walk(main_dir):
    _.sort()
    for filename in (filenames):
        files.append(os.path.join((dirname), filename))

files=[x for x in files if '.csv' not in x]

train_images= [(x) for x in files if 'train'  in x]
test_images = [(x) for x in files if 'test'   in x]

