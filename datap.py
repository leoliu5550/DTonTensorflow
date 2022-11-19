import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

dataset = tf.data.Dataset.list_files('./data/train/images/*', shuffle=False)
class_names = ['cat', 'deer', 'dog', 'fox', 'person', 'rabbit', 'raccoon']

def get_label(file_path):
    path = b'data/train/labels/'
    parts = tf.strings.split(file_path, os.path.sep)[-1]
    parts = tf.strings.split(parts,'.jpg')[-2]
    label_path = tf.strings.join([path,parts,b'.txt' ])
    return label_path
# pa = "data/train/images/000037-shed-cat_jpg.rf.33c39eb2d076a7485c48ce8ae8683791.jpg"
# print(get_label(pa))
# print(tf.io.gfile.exists(get_label(pa)))
# ans = get_label("data/train/images/000037-shed-cat_jpg.rf.33c39eb2d076a7485c48ce8ae8683791.jpg")

# print(tf.io.gfile.exists(ans.numpy()))

def process_image(file_path):

    img = tf.io.read_file(file_path) # load the raw data from the file as a string
    img = tf.image.decode_jpeg(img)
    label_path = get_label(file_path)
    label = tf.io.read_file(label_path)
    return img, label

# img, label = process_image(b"data/train/images/000037-shed-cat_jpg.rf.33c39eb2d076a7485c48ce8ae8683791.jpg")
train_dataset = dataset.map(process_image)
for image, label in train_dataset.take(1):
    print("****image",image)
    print("****label",label)

# plt.imshow(img)
# plt.show()
