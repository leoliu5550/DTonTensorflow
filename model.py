import tensorflow as tf 
import numpy as np
import logging 
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings("ignore")
# logger setup

# logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.DEBUG)
# experimentfile = 'exp'+str(len(os.listdir("./experiment")))
# log_path = os.path.join("./experiment",experimentfile)
# time_line = time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
# logfile  = time_line + '.log'
# os.mkdir(log_path)
# logfile = log_path+'/'+ logfile
# handler = logging.FileHandler(logfile,mode='w')
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter(
#     "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

# handler.setFormatter(formatter)
# console = logging.StreamHandler()
# console.setLevel(logging.WARNING)
# logger.addHandler(handler)
# logger.addHandler(console)

# logger.debug('This is a debug message.')
# logger.info('This is an info message.')
# logger.warning('This is a warning message.')
# logger.error('This is an error message.')
# logger.critical('This is a critical message.')

dataset = tf.data.Dataset.list_files('./data/train/images/*', shuffle=False)
# print(len(dataset))
# print(type(dataset))

# for file in dataset.take(3):
#     print(file.numpy())

# images_ds = dataset.shuffle(200)
# for file in images_ds.take(3):
#     print(file.numpy())
class_names = ['cat', 'deer', 'dog', 'fox', 'person', 'rabbit', 'raccoon']


def get_label(file_path):
    import os
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

print(get_label("./data/train/images/000037-shed-cat_jpg.rf.33c39eb2d076a7485c48ce8ae8683791.jpg"))
def process_image(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path) # load the raw data from the file as a string
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])
    return img, label