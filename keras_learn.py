#!python3
# -*- coding: utf-8

import os, sys
import re
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import SGD, Adagrad, Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = []
Y = []

x_size = 64
y_size = 64
lr_param = 0.001
moment=0.09
cat_dir = './training_set/cats/'
dog_dir = './training_set/dogs/'

test_catdir = './test_set/cats/'
test_dogdir = './test_set/dogs/'

# ファイル名の取得
def list_pictures(directory, ext='jpg|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.([\w]*\.)*(?:' + ext + '))', f.lower())]

def image_to_input_data(images, X, Y, num, padding=False):
  # global X
  # global Y
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

print("\n Pre load Success.")

# フォルダの中にある画像を順次読み込む
# カテゴリーは0から始める

X_train = []
y_train = []

X_test = []
y_test = []

# 対象Aの画像
cat_image = list_pictures(cat_dir, 'jpg')
image_to_input_data(cat_image, X_train, y_train, 0)
# 対象Bの画像
dog_image = list_pictures(dog_dir, 'jpg')
image_to_input_data(dog_image, X_train, y_train, 1)

catimage = list_pictures(test_catdir, 'jpg')
image_to_input_data(catimage, X_test, y_test, 0)
dogimage = list_pictures(test_dogdir, 'jpg')
image_to_input_data(dogimage, X_test, y_test, 1)


print("\n Image load Success.")

# arrayに変換
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_trest = np.asarray(y_test)

# 画素値を0から1の範囲に変換
X_train = X_train.astype('float32')
X_train = X_train / 255.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0

# クラスの形式を変換
# Y = np_utils.to_categorical(Y, 2)
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

# 学習用データとテストデータ
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=111)

print("\n Dataset setting Success.")

# CNNを構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))       # クラスは2個
model.add(Activation('softmax'))

print("\n model Summary\n")

model.summary()

print("\n Compile Start!!\n")

if len(sys.argv) >= 2:
  # データのロード
  model.load_weights('./checkpoint/my_checkpoint')

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=lr_param, momentum=moment),
              metrics=['accuracy'])

# 実行。出力はなしで設定(verbose=0)。
history = model.fit(X_train, y_train, batch_size=8, epochs=90,
                   validation_data = (X_test, y_test), verbose = 1)

# 重みの保存
model.save_weights('./checkpoint/my_checkpoint')

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
# 学習履歴の表示
# print(history.history['accuracy'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'], loc='lower right')
plt.show()

print("\n learning finish.")
