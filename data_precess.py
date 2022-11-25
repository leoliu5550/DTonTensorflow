import tensorflow as tf
import cv2
import os
import numpy as np

# dataset = tf.data.Dataset.list_files('./data/train/images/*', shuffle=False)
class_names = ['cat', 'deer', 'dog', 'fox', 'person', 'rabbit', 'raccoon']
"""
class_names2 = {
    0:'cat',
    1:'deer', 
    2:'dog', 
    3:'fox',
    4:'person',
    5:'rabbit',
    6:'raccoon'}
"""
IMG_PATH =["data/train/images","data/valid/images","data/test/images"]
#create TFRECORD 
train_dataset = tf.data.Dataset.list_files(IMG_PATH[0]+'/train/A/*.jpg')

def get_label_path(img_file_path):
    path = b'labels'
    # get file id like '0a2c3fdb-6788-4217-9a3e-fa1c9fbb18a4-dog_jpg.rf.9509acdcace6ac82c679652c058d168f'
    parts = tf.strings.split(img_file_path, os.path.sep)[-1]
    parts = tf.strings.split(parts,'.jpg')[-2]
    # get parents folder brfore id
    sple = tf.strings.split(img_file_path, os.path.sep)[:-2]
    parents=b''
    for i in range(sple.shape[0]):
        parents = tf.io.gfile.join(parents,sple[i].numpy())
    label_path = tf.strings.join([parts,b'.txt'])
    label_path = tf.io.gfile.join(parents,path,label_path.numpy())
    return label_path
# fixed this 
def process_image(file_path):
    img = tf.io.read_file(file_path) # load the raw data from the file as a string
    img = tf.image.decode_jpeg(img)
    label_path = get_label_path(file_path)
    #replace label = tf.io.read_file(label_path)
    with open(label_path, 'r') as bboxfile:
        records = bboxfile.readlines()
    labels = [int(lab.split(" ")[0]) for lab in records]
    """
    Yolo 格式 x, y, w, h
    - x, y 代表該bndBox的中心座標與圖片寬高的比值,是bndBox歸一化後的中心座標
    - w, h代表該bndBox的寬高與輸入圖像寬高的比值,是bndBox歸一化後的寬高座標
    """
    x = [float(x.split(" ")[1]) for x in records]
    y = [float(y.split(" ")[2]) for y in records]
    w = [float(w.split(" ")[3]) for w in records]
    h = [float(h.split(" ")[3]) for h in records]
    bbox_list = tf.convert_to_tensor(np.array([labels, x, y, w, h] ))
    return img, bbox_list

# train_dataset.map(process_image)