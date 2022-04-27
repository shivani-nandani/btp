from torchreid.utils import FeatureExtractor
import torch.nn as nn
import sklearn.metrics
import os
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
import numpy as np


def torchreidReid(SIMILARITY_THRESH, PATH_TO_DATA, PATH_TO_OUTPUT='.', FRAME_LIM=0, ALL_FRAMES=1):
    """re-identification task using torchreid library

    Args:
        SIMILARITY_THRESH (float): minimum similarity for object to be re-identified 
        PATH_TO_DATA (string): path to data for re-identification task 
        PATH_TO_OUTPUT (_type_): path to save the output (re-identification results csv). Defaults to current directory.
        FRAME_LIM (int, optional): for testing purposes change this to number of frames you want to proces. Defaults to 0.
        ALL_FRAMES (int): if 1 checks all previous frames for reid, if 0 checks only the previous frame. Defaults to 1.

    """

    # model to be used to extract features
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='model_weights/osnet_x1_0_imagenet.pth',
        device='cuda'
    )

    # cosine similarity function
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # file name format: 'frame<frame#>_<objID>.jpg'
    def split_name(filename):
        x = filename.split("_")
        return x[0].replace('frame', ''), x[1].replace('.jpg', '')

    filenames = []
    # sort the files in the order of frames
    for filename in os.listdir(PATH_TO_DATA):
        frame_no, obj_id = split_name(filename)
        num = int(frame_no + obj_id)
        filenames.append([filename, num])
    filenames.sort(key=lambda x: x[1])

    # dict format: {frame# : ['frame<frame#>_0.jpg', 'frame<frame#>_1.jpg', ...]}
    frame_dict = {}
    for filename, _ in filenames:
        frame_no, obj_id = split_name(filename)
        frame_no = int(frame_no)
        if FRAME_LIM and frame_no > FRAME_LIM:
            break
        try:
            frame_dict[frame_no].append(filename)
        except KeyError as e:
            frame_dict[frame_no] = [filename]

    # dict format: {frame# : [[frame<frame#>_0.jpg, features, objID], [frame<frame#>_1.jpg, features, objID], ...]}
    frames = dict.fromkeys(range(len(frame_dict)))
    # running count of unique objects counted
    obj_id = 0
    for i, [frame_no, filenames_lst] in enumerate(tqdm(frame_dict.items())):
        # inside a frame
        obj_lst = []
        for filename in filenames_lst:
            # inside a file belonging to the frame
            obj = [filename, extractor('street/street_persons_frames/' + filename)]
            if frame_no == 0:
                # all objects will be new
                obj.append(obj_id)
                obj_id+=1
            else:
                if ALL_FRAMES:
                    # check all the previous frames
                    max_sim = 0
                    for f in range(frame_no):
                        for past_frame_obj in frames[f]:
                            sim = cos(past_frame_obj[1],obj[1])
                            if sim >= SIMILARITY_THRESH:
                                if sim >= max_sim:
                                    max_sim = sim
                                    obj_id_to_append = past_frame_obj[2]
                    if max_sim != 0:
                        obj.append(obj_id_to_append)
                else:
                    # check only the previous frame
                    # check the previous frame if the object has already been identified
                    for prev_frame_obj in frames[frame_no-1]:
                        if cos(prev_frame_obj[1],obj[1]) >= SIMILARITY_THRESH:
                            # same object
                            obj.append(prev_frame_obj[2])
                if len(obj) == 2:
                    # no matches from the previous/past frame/s
                    obj.append(obj_id)
                    obj_id += 1
            # add obj to obj_lst in the frames dict
            obj_lst.append(obj)
        frames[frame_no] = obj_lst   

    # dict_file = open('street_ReID_v2.pkl', 'wb')
    # pickle.dump(frames, dict_file)
    # dict_file.close()
    # print('\ndict dumped')

    reid = []
    for frame, obj_lst in frames.items():
        for obj in obj_lst:
            reid.append([frame,obj[0],obj[2]])

    reid_df = pd.DataFrame(reid, columns=['frame#', 'filename', 'id'])
    reid_df.to_csv(PATH_TO_OUTPUT + '/reid.csv')
