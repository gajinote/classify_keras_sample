#!python3
# -*- coding: utf-8

import os, glob
import re
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import SGD, Adagrad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

# ファイル名の取得
def list_pictures(directory, ext='jpg|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.([\w]*\.)*(?:' + ext + '))', f.lower())]

print("\n Pre load Success.")

# フォルダの中にある画像を順次読み込む
# カテゴリーは0から始める

X = []
Y = []

x_size = 64
y_size = 64
lr_param = 0.005
moment=0.01
dog_dir = './dog/'
cat_dir = './cat/'

# 対象Aの画像
cat_image = list_pictures(cat_dir, 'jpg')
# print(cat_image)
for picture in cat_image:
    img = img_to_array(load_img(picture, target_size=(x_size, y_size)))
    X.append(img)

    Y.append(0)

# 対象Bの画像
dog_image = list_pictures(dog_dir, 'jpg')
# print(dog_image)
for picture in dog_image:
    img = img_to_array(load_img(picture, target_size=(x_size, y_size)))
    X.append(img)

    Y.append(1)

print("\n Image load Success.")

# arrayに変換
X = np.asarray(X)
Y = np.asarray(Y)

# 画素値を0から1の範囲に変換
X = X.astype('float32')
X = X / 255.0

# クラスの形式を変換
# Y = np_utils.to_categorical(Y, 2)
Y = to_categorical(Y, 2)

# 学習用データとテストデータ
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

print("\n Dataset setting Success.")

# CNNを構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))       # クラスは2個
model.add(Activation('softmax'))

# データのロード
model.load_weights('./checkpoint/my_checkpoint')

# テストデータ10件の正解ラベル
y_label = np.argmax(y_test[0:30], axis = 1)
# true_classes = np.argmax(y_test[0:10], axis = 1)
true_classes = []
for i in y_label:
  label = ""
  if i == 0:
    label= "cat"
  else:
    label = "dog"
  true_classes.append(label)


# テストデータの予測ラベル
x_label = np.argmax(model.predict(X_test[0:30]), axis=1)
pred_class = []
for i in x_label:
  label = ""
  if i == 0:
    label= "cat"
  else:
    label = "dog"
  pred_class.append(label)

pred_probs = np.max(model.predict(X_test[0:30]), axis=1)
pred_probs = ['{:.4f}'.format(i) for i in pred_probs]

# テストデータの画像と正解ラベルを出力
plt.figure(figsize=(16, 6))
for i in range(30):
  plt.subplot(3, 10, i+1)
  plt.axis("off")
  if pred_class[i] == true_classes[i]:
    plt.title(true_classes[i] + '\n' + pred_probs[i])
  else:
    plt.title(true_classes[i] + '\n' + pred_probs[i], color = "red")
  plt.imshow(X_test[i])
plt.show()
print("\n evaluate finish.")
