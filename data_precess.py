import tensorflow as tf
import cv2
import os
import numpy as np

# dataset = tf.data.Dataset.list_files('./data/train/images/*', shuffle=False)
HEIGHT =608
WIDTH = 608

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
train_dataset = tf.data.Dataset.list_files(IMG_PATH[0]+'/*.jpg')

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
    filename = file_path
    img = tf.io.read_file(file_path) # load the raw data from the file as a string
    img = tf.image.decode_image(img)   
    img = tf.image.encode_jpeg(img) 
    # height = img.shape[0]
    # width  = img.shape[1]
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
    # label_data={"image":img, "height":height, "width":width, "bbox":bbox_list, "filename":filename}
    # return label_data
    return img,bbox_list,filename

# 產生出TFrecord的格式資料
def make_example(*args):
    
    """ image, height, width, bbox, filename """
    args = args[0]
    image, bbox, filename = args[0],args[1],args[2]
    colorspace = b'RGB'
    channels = 3
    img_format = b'JPEG'
    # print(type(image.numpy()))
    data =  tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'height' : tf.train.Feature(int64_list=tf.train.Int64List(value=[HEIGHT])),
                    'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[WIDTH])),
                    'channels' : tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
                    'colorspace' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[colorspace])),
                    'img_format' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format])),
                    'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[0])),
                    'bbox_xmin' : tf.train.Feature(float_list=tf.train.FloatList(value=bbox[1])),
                    'bbox_xmax' : tf.train.Feature(float_list=tf.train.FloatList(value=bbox[2])),
                    'bbox_ymin' : tf.train.Feature(float_list=tf.train.FloatList(value=bbox[3])),
                    'bbox_ymax' : tf.train.Feature(float_list=tf.train.FloatList(value=bbox[4])),
                    'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename]))
                }
            )
        )
    return data

data = b"data/test/images/647617de-e5d3-4f65-adb1-ead45b3b76be-raccoon_jpg.rf.67add1362c925b75c8843810e42b7f97.jpg"
# print(type(process_image(data)))
# print(process_image(data))
print(make_example(process_image(data)))


 
#这个函数用来生成TFRECORD文件，第一个参数是列表，每个元素是图片文件名，第二个参数是写入的目录名
#第三个参数是文件名的起始序号，第四个参数是队列名称，用于和父进程发送消息
# dataset = dataset.map(process_image)

# for da in dataset.take(2):
#     print(da)


# def int64_feature(value):
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# # 轉Bytes資料為 tf.train.Feature 格式
# def bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def convert_to_TFRecord(images, labels, filename):
#     n_samples = len(labels)
#     TFWriter = tf.python_io.TFRecordWriter(filename)

#     print('\nTransform start...')
#     for i in np.arange(0, n_samples):
#         try:
#             image = cv2.imread(images[i])
#             image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#             height,width,channel = image.shape
#             image = cv2.resize(image,(224,224))
            
#             if image is None:
#                 print('Error image:' + images[i])
#             else:
#                 image_raw = image.tostring()

#             label = int(labels[i])
            
#             # 將 tf.train.Feature 合併成 tf.train.Features
#             ftrs = tf.train.Features(
#                     feature={'Label': int64_feature(label),
#                              'image_raw': bytes_feature(image_raw),
#                              'channel':int64_feature(channel),
#                              'width':int64_feature(width),
#                              'height':int64_feature(height)}
#                    )
        
#             # 將 tf.train.Features 轉成 tf.train.Example
#             example = tf.train.Example(features=ftrs)

#             # 將 tf.train.Example 寫成 tfRecord 格式
#             TFWriter.write(example.SerializeToString())
#         except IOError as e:
#             print('Skip!\n')

#     TFWriter.close()
#     print('Transform done!')