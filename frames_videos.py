from json import detect_encoding
import cv2
import numpy as np
import os
import shutil

from os.path import isfile, join


def videoToFrames(input_path):
    """Converts video to frames from the given path

    Args:
        input_path (string): path to the video to be converted 

    Returns:
        string : output path
    """

    output_path = './frames'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)           # Removes all the subdirectories
        os.makedirs(output_path)

    # Opens the Video file
    cap = cv2.VideoCapture(input_path)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('./frames/frame'+str(i)+'.jpg', frame)
        i += 1

    cap.release()

    return output_path


def framesToVideo(fps, input_path=None, output_path=None):
    """Converts frames to video (avi format only)

    Args:
        output_path (string) : path (with name) to save the final video 
        fps (float) : frames per second in the converted video
        input_path (string) : directory path to frames, Default = None         

    Returns:
        string : output path
    """

    if input_path != None:
        input_path = input_path + '/'
        files = [f for f in os.listdir(
            input_path) if isfile(join(input_path, f))]

        # for sorting the file names properly
        files.sort(key=lambda x: int(x[5:-4]))
        # print(files)

        filenames = [input_path + files[i] for i in range(len(files))]

    else:
        filenames = ['./runs/detect/'+subd+'/'+str(os.listdir('./runs/detect/'+subd)[
                                                   0]) for subd in os.listdir('./runs/detect')]
        filenames.sort(key=lambda x: int(x.split('/')[-1][5:-4]))

    frame_array = []

    for filename in filenames:
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # print(filename)

        # inserting the frames into an image array
        frame_array.append(img)

    if output_path != None:
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    else:
        out = cv2.VideoWriter(
            'detected.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

    return output_path
