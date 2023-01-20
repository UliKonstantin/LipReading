import cv2
import numpy as np
from PIL import Image as im
import os


def vid2pics(video_path, output_dir):
    # Open the video file
    if not os.path.exists(video_path):
        print("file doesn't exist: %s"%video_path)
        return
    video = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the frames from the video and save them as images
    frame_count = 0
    while True:
        # Read the next frame from the video
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        crop_img = frame[105:201, 79:175]
        # If the frame was not read successfully, we have reached the end of the video
       
        # Save the frame as an image
        cv2.imwrite(os.path.join(output_dir, "frame%.2d.jpg"%frame_count), crop_img)
        frame_count += 1

    # Release the video file
    video.release()


 



def pre_wraper(source_dir):
    output_path = "proc_data"

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    lable_list = os.listdir(source_dir)

    for lable in lable_list:
        print("Working on: '%s'"%lable)
        set_list = os.listdir(source_dir+"/"+lable)
        if not os.path.exists(output_path+"/"+lable):
            os.makedirs(output_path+"/"+lable, exist_ok=True)

        for group in set_list:
            directory_path = source_dir+"/"+lable+"/"+group
            if not os.path.exists(output_path+"/"+lable+"/"+group):
                os.makedirs(output_path+"/"+lable+"/"+group, exist_ok=True)

            for filename in os.listdir(directory_path):
                              
                if filename.endswith(".mp4"):
                    if not os.path.exists(output_path+"/"+lable+"/"+group+"/"+filename):
                        os.makedirs(output_path+"/"+lable+"/"+group+"/"+filename, exist_ok=True)
                    file_path = os.path.join(directory_path, filename)
                    vid2pics(file_path,output_path + "/"+lable+ "/"+group+ "/"+filename) # need to distiguish between lable and groups




pre_wraper("data")


     
    


