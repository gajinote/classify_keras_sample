#!python3

from tensorflow.python.keras.preprocessing.image import img_to_array
import os, sys, re
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        channel_shift_range=100.0,
        fill_mode='nearest')

cat_dir = './training_set/cats/'
dog_dir = './training_set/dogs/'

# ファイル名の取得
def list_pictures(directory, ext='jpg|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.([\w]*\.)*(?:' + ext + '))', f.lower())]

def image_to_input_data(images, dir_name, type=0):
  if (type == 0): pref_string = 'cat-1'
  else: pref_string = 'dog-1'
  for picture in images:
    img = load_img(picture)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=2,
                              save_to_dir='preview', save_prefix=pref_string, save_format='jpg'):
      i += 1
      if i > 2:
        break
  
# CatImageの処理
cat_images = list_pictures(cat_dir, 'jpg')
image_to_input_data(cat_images, dir_name='previw', type=0)
dog_images = list_pictures(dog_dir, 'jpg')
image_to_input_data(dog_images, dir_name='previw', type=1)
