#%%
import tensorflow as tf
import pandas as pd 

#%%
emo = input("emotion:")
emo_data = pd.read_csv("data/"+emo+"_data.csv")
emo_data

