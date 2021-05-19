#!python3
# -*- coding: utf-8

import os, glob
import re
import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ファイル名の取得
def list_pictures(directory, ext='jpg|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

print("\n Pre load Success.")

# フォルダの中にある画像を順次読み込む
# カテゴリーは0から始める

X = []
Y = []

# 対象Aの画像
cat_image = list_pictures('./cat/', 'jpg')
# print(cat_image)
for picture in cat_image:
    img = img_to_array(load_img(picture, target_size=(64,64)))
    X.append(img)

    Y.append(0)

# 対象Bの画像
dog_image = list_pictures('./dog/', 'jpg')
# print(dog_image)
for picture in dog_image:
    img = img_to_array(load_img(picture, target_size=(64,64)))
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
Y = np_utils.to_categorical(Y, 2)

# 学習用データとテストデータ
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=111)

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

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# 実行。出力はなしで設定(verbose=0)。
history = model.fit(X_train, y_train, batch_size=5, epochs=200,
                   validation_data = (X_test, y_test), verbose = 1)

# 学習履歴の表示
# print(history.history['accuracy'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()

# テストデータに適用
# predict_classes = model.predict_classes(X_test)
predict_classes = np.argmax(model.predict(X_test), axis=-1)

# マージ。yのデータは元に戻す
mg_df = pd.DataFrame({'predict': predict_classes, 'class': np.argmax(y_test, axis=1)})

# confusion matrix
pd.crosstab(mg_df['class'], mg_df['predict'])

loss, acc = model.evaluate(X_test, y_test, verbose=2)
print("model accuracy: {:5.2f}%".format(100*acc))

# print(mg_df['class'], mg_df['predict'])

# 重みの保存
model.save_weights('./checkpoint/my_checkpoint')

print("\n learning finish.")