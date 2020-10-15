import pandas as pd 
import numpy as np
import category_encoders as ce
csv=("/home/tkrsh/osic-main/train.csv")

df=pd.read_csv(csv)

cat_features=['Age','Sex','SmokingStatus']
train_df=df.drop("Patient",axis=1)

target_enc = ce.CatBoostEncoder(cols=cat_features)
target_enc.fit(train[cat_features], df['FVC'])


train_CBE = train.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))