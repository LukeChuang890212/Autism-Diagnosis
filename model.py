#%%
import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#引入keras
from tensorflow import keras
#以下為做DNN常引入的重要模組
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

#%%
emo = input("emotion:")
emo_data = pd.read_csv("data/"+emo+"_data.csv").drop(["Unnamed: 0", "emotion"], axis=1)
emo_data

features = emo_data.iloc[:,:-1]
targets = emo_data.iloc[:,-1]

features
targets

print("number of normal data:{}".format(len(emo_data[emo_data.group == 0])))
print("number of autism data:{}".format(len(emo_data[emo_data.group == 1])))

#%%
# 分割測試、訓練資料
train_x,test_x,train_y,test_y = train_test_split(
    features,      #放特徵
    targets,      #放目標
    test_size = 0.3  #測試資料占3成
    )

print("number of normal train_data:{}".format(len(train_y[train_y == 0])))
print("number of autism train_data:{}".format(len(train_y[train_y == 1])))

#%%
#模型初始化
model = Sequential()
#模型建構
model.add(Dense(100,activation="relu",input_dim=3))
model.add(Dense(100,activation="relu"))
model.add(Dense(100,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

#%%
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

#%%
model.summary()

#%%
# Early stop
early_stopping = EarlyStopping(
    monitor = 'val_accuracy',
    min_delta = 0.01,
    patience = 300,
    verbose = 1,
    mode = "max"
)

#%%
def fit(model, train_x, train_y, early_stopping):
    history = model.fit(train_x,train_y,    #輸入與輸出
            epochs = 10000,      #迭代數
            verbose = 1,      #顯示訓練過程
            validation_split = 0.2, #驗證資料分割比例
            callbacks = [early_stopping]
                        )
    return history

#%%
def eval(model, train_x, train_y, test_x, test_y):
    train_result = model.evaluate(train_x,train_y)
    test_result = model.evaluate(test_x,test_y)
    print("-----------")
    print("train loss:{},train acc:{}".format(train_result[0],train_result[1]))
    print("test loss:{},test acc:{}".format(test_result[0],test_result[1]))

    return train_result[1]

#%%
num_autism_train = len(train_y[train_y == 1])
current_train_acc = 0
train_acc = 1
fit_time = 0
history = None
while(train_acc-current_train_acc > 0.1):
    current_train_acc =train_acc

    train_y_tmp = shuffle(train_y[train_y == 0].sample(num_autism_train).append(train_y[train_y == 1]))
    train_x_tmp = train_x.loc[train_y_tmp.index]

    history = fit(model, train_x_tmp, train_y_tmp, early_stopping)
    train_acc = eval(model, train_x_tmp, train_y_tmp, test_x, test_y)

    fit_time += 1
    print("-----------")
    print("fit time:{}".format(fit_time))

#%%
plt.subplot(211)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel("accurary")
plt.title("accurary history")
plt.legend(['train_acc','test_acc'],loc='upper left')

plt.subplot(212)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel("loss")
plt.title("loss history")
plt.legend(['train_loss', 'test_loss'],loc='upper left')
plt.xlabel("epoch")

plt.tight_layout()
plt.show()

#%%
predict_y = model.predict(test_x)
predict_class_y = []
for proba in predict_y:
  if(proba > 0.5): #閾值
    predict_class_y.append(1)
  else:
    predict_class_y.append(0)
predict_class_y = np.array(predict_class_y).reshape(-1)

#%%
confusion = pd.crosstab(test_y,predict_class_y,rownames=['true'],colnames=['predict'])
confusion