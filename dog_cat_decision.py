#!python3
# -*- coding: utf-8

import os, glob
import re
import cv2, tkinter
from tkinter import ttk
from tkinter import filedialog
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

# ファイル読み込み
def load_file():
  fType = [("静止画ファイル", "*.jpg;*.png, *.jpeg")]
  iDir = os.path.abspath(os.path.dirname(__file__))
  path = filedialog.askopenfilename(filetypes=fType, initialdir=iDir)
  file_path.set(path)

# 判定
def exec_decisiotn():
  path = file_path.get()
  print(path)

X = []
Y = []

x_size = 64
y_size = 64
print("\n Pre load Success.")

# CNNを構築
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64, 64, 3)))
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

# メインウィンドウ
main_win = tkinter.Tk()
main_win.title("Dog/Cat/Descriminator")
main_win.geometry("640x240")

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

# ウィジェット配置
folder_label.grid(column=0, row=0, pady=10)
folder_box.grid(column=1, row=0, columnspan=4, sticky=tkinter.EW, padx=5)
folder_btn.grid(column=5, row=0)

exec_btn.grid(column=1, row=1)

main_win.mainloop()
