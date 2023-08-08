import os
import time

label_dic = {0: 'flow', 1: 'drop'}
device = 1

root_dir = 'D:/hm/Projoects/yolo8/Storage_temp/Data/Image/TM_Leak/MRCNN'
train_raw_image_path = root_dir + '/train'
test_raw_image_path = root_dir + '/test'

img_size = 1280
batch_size = 8
epoch = 500

train_save_path = 'D:/hm/Projoects/yolo8/runs/segment/weights/train_seg2'
model_path = train_save_path + '/weights/best.pt'
val_save_path = train_save_path+'/val'
test_save_path = train_save_path+'/test'

copy_data_path = 'D:/hm/Projoects/yolo8/data/images'
train_data_path = copy_data_path + '/train'
test_data_path = copy_data_path + '/test'
val_data_path = copy_data_path + '/val'

yaml_path = 'D:/hm/Projoects/yolo8/TM_yolov8_seg.yaml'
