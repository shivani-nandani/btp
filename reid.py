from torchreid.utils import FeatureExtractor
import torch.nn as nn
import sklearn.metrics
import os
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
import numpy as np


def torchreidReid(SIMILARITY_THRESH, PATH_TO_DATA, PATH_TO_OUTPUT='.', FRAME_LIM=0):
    """re-identification task using torchreid library

    Args:
        SIMILARITY_THRESH (float): minimum similarity for object to be re-identified 
        PATH_TO_DATA (string): path to data for re-identification task 
        PATH_TO_OUTPUT (_type_): path to save the output (re-identification results csv). Defaults to current directory.
        FRAME_LIM (int, optional): for testing purposes change this to number of frames you want to proces. Defaults to 0.

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

    # where the magic happens
    # dict format: {frame# : [[frame<frame#>_0.jpg, features, objID], [frame<frame#>_1.jpg, features, objID], ...]}
    frames = dict.fromkeys(range(len(frame_dict)))
    # running count of unique objects counted
    obj_id = 0
    for i, [frame_no, filenames_lst] in enumerate(tqdm(frame_dict.items())):
        # inside a frame
        obj_lst = []
        for filename in filenames_lst:
            # inside a file belonging to the frame
            obj = [filename, extractor(PATH_TO_DATA + '/' + filename)]
            obj_lst.append(obj)
        if frame_no == 0:
            # all objects will be new
            for obj in obj_lst:
                obj.append(obj_id)
                obj_id += 1
        else:
            # compare frame features with previous frame features
            frame_features = [obj[1][0].cpu().numpy() for obj in obj_lst]
            prev_frame_features = [prev_obj[1][0].cpu().numpy()
                                   for prev_obj in frames[frame_no-1]]
            cos_sim = sklearn.metrics.pairwise.cosine_similarity(
                prev_frame_features, frame_features)

            # rank
            for col in range(len(cos_sim[0])):
                # get column
                scores_for_obj = [row[col] for row in cos_sim]
                # list of indexes of elements sorted in descending order
                sorted_indices = np.argsort(scores_for_obj)[::-1]
                # check the next max of col with those of its row
                for index in sorted_indices:
                    if scores_for_obj[index] > SIMILARITY_THRESH:
                        flag = 1
                        for check_col in range(len(cos_sim[0])):
                            if check_col != col:
                                if scores_for_obj[index] < cos_sim[index][check_col]:
                                    # not the max for the row therefore cannot assign this object id, move on to next element
                                    flag = 0
                        if flag:
                            # the max for the row, therefore assign that object id
                            obj_lst[col].append(frames[frame_no-1][index][2])
                # checked all valid elemenets and none suitable found, therefore assign a new id
                if len(obj_lst[col]) == 2:
                    obj_lst[col].append(obj_id)
                    obj_id += 1
        frames[frame_no] = obj_lst

    # dict_file = open('street/outputs/pkl/street_ReID.pkl', 'wb')
    # pickle.dump(frames, dict_file)
    # dict_file.close()

    reid = []
    for frame, obj_lst in frames.items():
        for obj in obj_lst:
            reid.append([frame, obj[0], obj[2]])

    reid_df = pd.DataFrame(reid, columns=['frame#', 'filename', 'id'])
    reid_df.to_csv(PATH_TO_OUTPUT + '/reid.csv')
