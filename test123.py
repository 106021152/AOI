import numpy as np
import os
import time
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
PATH = ""
data_path = PATH + 'E:/AOI/reAOIdata_test'
data_dir_list = os.listdir(data_path)    

img_data_list=[]

for dataset in data_dir_list :
  img_list = os.listdir(data_path+'/'+dataset)
  print('Loaded the images of dataset-'+dataset)
  for img in img_list:
    img_path = data_path + '/' + dataset + '/' +img
  
    ### END CODE HERE ###
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)
    #讀取圖片
    #將讀取到的圖片轉成array
    #在陣列最前面增加一個維度
    #進行標準化
    #把標準化的值存入list
		### START CODE HERE ###  (≈ 5 lines)
    
    ### END CODE HERE ###

img_data = np.array(img_data_list)

print (img_data.shape)  #(600, 1, 224, 224, 3)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)  #(1, 600, 224, 224, 3)
img_data=img_data[0]
print (img_data.shape)  #(600, 224, 224, 3)

# 給每張圖片標上Label
### START CODE HERE ###  (≈ 3 lines)
num_classes = 6
num_of_samples = img_data.shape[0]
print(num_of_samples)
labels = np.ones((num_of_samples,),dtype='int64')
### END CODE HERE ###

labels[0:674]=0
labels[674:1166]=1
labels[1166:1266]=2
labels[1266:1644]=3
labels[1644:1884]=4
labels[1884:2528]=5

# one-hot encoding
###  CODE HERE ###  (≈ 1 lines)
Y = np_utils.to_categorical(labels, num_classes)    
#將圖片與Label對應
###  CODE HERE ###  (≈ 1 lines)
x,y = shuffle(img_data,Y, random_state=2)
# 切割資料分為訓練(train)跟測試(test)
###  CODE HERE ###  (≈ 1 lines)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)



#設定input shape
###  CODE HERE ###  (≈ 1 lines)
image_input = Input(shape=(224,224,3))
#取用VGG16模型 include_top為是否保留最上層三層的全連接層
#include_top=ture 默認輸入為(224,224,3) 須調整則須改成 False
###  CODE HERE ###  (≈ 1 lines)
model = VGG16(input_tensor=image_input, include_top = True,weights='imagenet')
#輸出模型層數概況
###  CODE HERE ###  (≈ 1 lines)
model.summary()
last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

for layer in custom_vgg_model.layers[:-1]:
	layer.trainable = False

#設定模型的損失函數 優化器 列出準確率
###  CODE HERE ###  (≈ 1 lines)
custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

t=time.time()
#輸入訓練模型需要的參數
###  CODE HERE ###  (≈ 1 lines)
hist = custom_vgg_model.fit(X_train, y_train, batch_size=32,epochs=10,verbose=1,validation_data=(X_test,y_test))
print('Training time: %s' % time.time())

#評估訓練完的模型對測試資料的準確度
###  CODE HERE ###  (≈ 1 lines)
(loss, accuracy)=custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose = 1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#預測測試資料
#印出預測出的Label值

### START CODE HERE ###  (≈ 5 lines)
prediction = custom_vgg_model.predict(X_test)
predict = np.argmax(prediction,axis=1)
ans=np.argmax(y_test,axis=1)
print(predict)
print(ans)
    ### END CODE HERE ###

PATH = ""
data_path = PATH + 'E:/AOI/reAOIdata_test'
data_dir_list = os.listdir(data_path)    

img_data_list=[]

for dataset in data_dir_list :
  img_list = os.listdir(data_path+'/'+dataset)
  print('Loaded the images of dataset-'+dataset)
  for img in img_list:
    img_path = data_path + '/' + dataset + '/' +img
  
    ### END CODE HERE ###
    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)
    #讀取圖片
    #將讀取到的圖片轉成array
    #在陣列最前面增加一個維度
    #進行標準化
    #把標準化的值存入list
		### START CODE HERE ###  (≈ 5 lines)
    
    ### END CODE HERE ###

img_data = np.array(img_data_list)

print (img_data.shape)  #(600, 1, 224, 224, 3)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)  #(1, 600, 224, 224, 3)
img_data=img_data[0]
print (img_data.shape)  #(600, 224, 224, 3)

prediction = custom_vgg_model.predict(img_data)
predict =np.argmax(prediction,axis=1)
print(predict)

model.save("E:/AOI/model_test.h5")

import pandas as pd
df_test = pd.read_csv("test.csv")
df_test['Label'] = predict
df_test.to_csv("answer.csv",index=False)
