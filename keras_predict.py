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

X = []
Y = []
X_L = []

x_size = 64
y_size = 64
lr_param = 0.005
moment=0.01
dog_dir = './dog/'
cat_dir = './cat/'

# ファイル名の取得
def list_pictures(directory, ext='jpg|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.([\w]*\.)*(?:' + ext + '))', f.lower())]

def image_to_input_data(images, X, Y, num, padding=False):
  # global X
  # global Y
  global X_L
  for picture in images:
    base_imga = load_img(picture)
    width, height = base_imga.size
    if padding == False or width == height:
      img = img_to_array(load_img(picture, target_size=(x_size, y_size)))
    
    elif width > height:
      aspect = height / width
      tgt_y = (int)(y_size * aspect)
      img = img_to_array(load_img(picture, target_size=(x_size, tgt_y)))
      padd_1 = (int)((y_size - tgt_y) / 2)
      padd_2 = (int)(y_size - tgt_y - padd_1)
      img = np.pad(img, [(0, 0), (padd_1, padd_2), (0,0)], 'constant')
    else:
      aspect = width / height
      tgt_x = (int)(x_size * aspect)
      img = img_to_array(load_img(picture, target_size=(tgt_x, y_size)))
      padd_1 = (int)((x_size - tgt_x) / 2)
      padd_2 = (int)(x_size - tgt_x - padd_1)
      img = np.pad(img, [(padd_1, padd_2), (0, 0), (0,0)], 'constant')
    X.append(img)
    Y.append(num)
    img = img_to_array(load_img(picture, target_size=(x_size*2, y_size*2)))
    X_L.append(img)

print("\n Pre load Success.")

# フォルダの中にある画像を順次読み込む
# カテゴリーは0から始める

# 対象Aの画像
cat_image = list_pictures(cat_dir, 'jpg')
image_to_input_data(cat_image, X, Y, 0, padding=False)
# 対象Bの画像
dog_image = list_pictures(dog_dir, 'jpg')
image_to_input_data(dog_image, X, Y, 1, padding=False)

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
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

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
model.load_weights('./80over/checkpoint/my_checkpoint')

# テストデータ10件の正解ラベル
y_label = np.argmax(y_test, axis = 1)
# true_classes = np.argmax(y_test[0:10], axis = 1)


# テストデータの予測ラベル
x_label = model.predict(X_test)

pred_probs = model.predict(X_test)
pred_probs = [['{:.4f}'.format(i), '{:.4f}'.format(j)]  for i, j in pred_probs]

# テストデータの画像と正解ラベルを出力
cnt = 0
color_r="black"
plt.axis("off")
plt.title("cat: " + pred_probs[cnt][0] + "\n" + "dog: " + pred_probs[cnt][1], color = color_r)
plt.imshow(X_test[cnt])
plt.show()
print("\n evaluate finish.")
