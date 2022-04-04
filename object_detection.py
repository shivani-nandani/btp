# object detection using YOLOv5
import os
import shutil
import torch
import numpy as np
import pandas as pd
from PIL import Image 


def loadModel(model_version):
    """Load the required model version

    Args:
        model_version (string): yolo5 version name to be loaded (yolov5n/yolov5s/yolov5m/yolov5l/yolov5x)

    Returns:
        torch model: model loaded from "ultralytics/yolov5"
    """

    # model
    model = torch.hub.load('ultralytics/yolov5', str(model_version), pretrained=True)#, force_reload=True)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
    return model



def objectDetection(model_version, frames_input_path, detection_results_path, total_frames):
    """Finds and anotates each image (frame) for all objects in the YOLOv5 class library

    Args:
        model_version (float): model version of the YOLOv5 used 
        frames_input_path (string): image path to frames 
        detection_results_path (string): path to save the detection results csv
        total frames (int): total number of frames 
    """

    model = loadModel(model_version=model_version)

    # dataframe to save detection results
    detection_results = pd.DataFrame(columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name', 'frame'])

    output_path = './runs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)           # Removes all the subdirectories
        os.makedirs(output_path)

    for i in np.arange(0,total_frames,1):

        image_path = frames_input_path+'/frame'+str(i)+'.jpg' 

        # Inference
        results = model(image_path)

        # Results 
        results.save()

        frame_obj_i_start = len(detection_results)
        detection_results = pd.concat([detection_results, results.pandas().xyxy[0]], ignore_index=True)
        frame_obj_i_end = frame_obj_i_start + len(results.pandas().xyxy[0])
        detection_results.loc[frame_obj_i_start:frame_obj_i_end, 'frame'] = i
        
    detection_results.to_csv(detection_results_path)



def cropPersons(detection_results_path, image_directory_path, cropped_image_output_path, confidence_threshold):
    """Crops individuals as per detection results for each frame 

    Args:
        detection_results_path (string): path to the detection results csv
        image_directory_path (_type_): path to the original frames/images (unanotated version)
        cropped_image_output_path (_type_): path to save the cropped images 
        confidence_threshold (_type_): threshold value that will filter images for reid based on confidence 
    """

    if not os.path.exists(cropped_image_output_path):
        os.makedirs(cropped_image_output_path)
    else:
        shutil.rmtree(cropped_image_output_path)           # Removes all the subdirectories
        os.makedirs(cropped_image_output_path)

    detection_results = pd.read_csv(detection_results_path)

    for i in range(len(detection_results)):

        if detection_results.name[i] == 'person' and detection_results.confidence[i]>=confidence_threshold:
            
            img = Image.open(image_directory_path+'/frame'+str(int(detection_results.frame[i]))+'.jpg') 

            left = detection_results.xmin[i]
            top = detection_results.ymin[i]
            right = detection_results.xmax[i]
            bottom = detection_results.ymax[i]

            img_res = img.crop((left, top, right, bottom))

            img_res.save(cropped_image_output_path+'/frame'+str(int(detection_results.frame[i]))+'_'+str(i)+'.jpg') 