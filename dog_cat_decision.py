#!python3
# -*- coding: utf-8

import os
import tkinter
from tkinter import ttk
from tkinter import filedialog
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt


# ファイル読み込み
def load_file():
  fType = [("静止画ファイル", "*.jpg;*.png, *.jpeg")]
  iDir = os.path.abspath(os.path.dirname(__file__))
  path = filedialog.askopenfilename(filetypes=fType, initialdir=iDir)
  file_path.set(path)

# 判定
def exec_decisiotn():
  global x_size
  global y_size
  global plt_show_flg
  path = file_path.get()
  print(path)
  img = img_to_array(load_img(path, target_size=(x_size, y_size)))
  X = []
  X.append(img)
  X = np.asarray(X)
  # 画素値を0から1の範囲に変換
  X = X.astype('float32')
  X = X / 255.0

  pred_probs = model.predict(X)  
  pred_probs = [['{:.4f}'.format(i), '{:.4f}'.format(j)]  for i, j in pred_probs]
  print("cat: " + pred_probs[0][0] + ", dog: " + pred_probs[0][1])
  # テストデータの画像と正解ラベルを出力
  plt_show_flg = 1
  color_r="black"
  plt.axis("off")
  plt.title("cat: " + pred_probs[0][0] + "\n" + "dog: " + pred_probs[0][1], color = color_r)
  plt.imshow(X[0])
  plt.show()

def clear_plt():
  global plt_show_flg
  if (plt_show_flg == 1):
    print("clear")
    plt.clf()
    plt.close()
  plt_show_flg = 0

X = []

plt_show_flg = 0

x_size = 128
y_size = 128
print("\n Pre load Success.")

# CNNを構築
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=(x_size, y_size, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
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
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))       # クラスは2個
model.add(Activation('softmax'))

# データのロード
model.load_weights('./checkpoint/my_checkpoint')

# メインウィンドウ
main_win = tkinter.Tk()
main_win.title("Dog/Cat/Descriminator")
main_win.geometry("480x100")

# メインフレーム
main_frm = ttk.Frame(main_win)
main_frm.grid(column=0, row=0, sticky=tkinter.NSEW, padx=5, pady=10)

# パラメータ
file_path = tkinter.StringVar()

# ウィジェット作成（フォルダパス）
folder_label = ttk.Label(main_frm, text="画像ファイル指定")
folder_box = ttk.Entry(main_frm, textvariable=file_path)
folder_btn = ttk.Button(main_frm, text="参照", command=load_file)
exec_btn = ttk.Button(main_frm, text="判定", command=exec_decisiotn)
close_btn = ttk.Button(main_frm, text="クリア",command=clear_plt)
blnk_label = ttk.Label(main_frm, text="         ")

# ウィジェット配置
folder_label.grid(column=0, row=0, pady=10)
folder_box.grid(column=1, row=0, columnspan=6, sticky=tkinter.EW, padx=2)
folder_btn.grid(column=7, row=0)

exec_btn.grid(column=1, row=1)
blnk_label.grid(column=2, row=1)
close_btn.grid(column=4, row=1)

main_win.mainloop()
