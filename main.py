## https://docs.ultralytics.com/tasks/segment/#export
## https://www.kaggle.com/code/nicolaasregnier/yolo-segmentation-on-grapes/notebook
## https://www.kaggle.com/code/stpeteishii/flickr-yolov8-object-detection-segmentation/notebook
## https://www.kaggle.com/code/mersico/medical-instance-segmentation-with-yolov8

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from tqdm import tqdm
import shutil as sh
import os
import pandas as pd
import tensorflow as tf
import keras as k
import subprocess
from tensorflow.keras.utils import img_to_array, load_img
import gc
import torch
import preprocessing as pp
import params
from ultralytics import YOLO
import ultralytics
from moviepy.editor import VideoFileClip

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.cuda.empty_cache()
torch.cuda.device(1)

if __name__ == '__main__':

    # model = YOLO('yolov8n-seg.yaml')  # build a new model from scratch
    # model = YOLO('yolov8n-seg.pt')
    #
    # print("Number OF Train Images:" + str(len(os.listdir(params.train_data_path))))
    # print("Number OF Test Images:" + str(len(os.listdir(params.val_data_path))))

    # model.train(data=params.yaml_path, name=params.train_save_path, epochs=params.epoch, imgsz=params.img_size)

    model = YOLO(params.model_path)  # load a custom model

    # model.val(data=params.yaml_path, name=params.val_save_path, batch=params.batch_size, imgsz=params.img_size)

    # for image in os.listdir(params.val_data_path):
    #     model.predict(data=params.yaml_path, source=params.val_data_path+'//'+image, name=params.test_save_path,
    #                   save=True, batch=params.batch_size,
    #                   imgsz=params.img_size)

    # video stream
    video_path = "D:/hm/Projoects/yolo8/data/video/30FPS/20220929_134008.avi"
    cap = cv2.VideoCapture(video_path)
    ###여기부터
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # fps = int(cap.get(5))
    #
    # output_path = "20220929_134008_seg.avi"
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    #
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     out.write(frame)
    #
    # cap.release()
    # out.release()
    #
    # # Convert the saved video to desired format (e.g., mp4)
    # output_format = "mp4"
    # output_final_path = "20220929_134008_seg." + output_format
    # clip = VideoFileClip(output_path)
    # clip.write_videofile(output_final_path, codec='libx264')
    # # Clean up
    # clip.close()

    ###여기까지가 저장

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
