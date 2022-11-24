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
    label = tf.io.read_file(label_path)



    label = tf.strings.split(label, sep="\n", maxsplit=-1)
    label = tf.strings.split(label, sep=" ", maxsplit=-1).to_tensor(default_value='')
    label = tf.strings.to_number(label,out_type=tf.dtypes.float32)
    return img, label


# should be start here
file_path = "data/train/images/0a480c46-4831-43ba-a5fc-08258c2b6f55-dog_person_jpg.rf.3d6102e59c01a16c9daf26977330e5d2.jpg"
label_path = get_label_path(file_path)
with open(label_path, 'r') as bboxfile:
    records = bboxfile.readlines()
    for record in records:
        fields = record.strip().split(',')
        print(fields)
        # filename = fields[0][:-4]
        # labels = [labels_dict[x] for x in fields[1].split(';')]
        # xmin = [float(x) for x in fields[2].split(';')]
        # ymin = [float(x) for x in fields[3].split(';')]
        # xmax = [float(x) for x in fields[4].split(';')]
        # ymax = [float(x) for x in fields[5].split(';')]
        # bbox_list[filename] = [labels, xmin, ymin, xmax, ymax] 
# label = tf.io.read_file(label_path)
# print(label)

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