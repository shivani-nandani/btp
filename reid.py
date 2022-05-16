from torchreid.utils import FeatureExtractor
import torch.nn as nn
import sklearn.metrics
import os
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
import warnings
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from operator import getitem

def get_euclidean(x, y, **kwargs):
    m = x.shape[0]
    n = y.shape[0]
    distmat = (
        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(1, -2, x, y.t())
    return distmat

def cosine_similarity(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """
    Computes cosine similarity between two tensors.
    Value == 1 means the same vector
    Value == 0 means perpendicular vectors
    """
    x_n, y_n = x.norm(dim=1)[:, None], y.norm(dim=1)[:, None]
    x_norm = x / torch.max(x_n, eps * torch.ones_like(x_n))
    y_norm = y / torch.max(y_n, eps * torch.ones_like(y_n))
    sim_mt = torch.mm(x_norm, y_norm.transpose(0, 1))
    return sim_mt

def get_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Computes cosine distance between two tensors.
    The cosine distance is the inverse cosine similarity
    -> cosine_distance = abs(-cosine_distance) to make it
    similar in behaviour to euclidean distance
    """
    sim_mt = cosine_similarity(x, y, eps)
    return torch.abs(1 - sim_mt).clamp(min=eps)

def get_dist_func(func_name="euclidean"):
    if func_name == "cosine":
        dist_func = get_cosine
    elif func_name == "euclidean":
        dist_func = get_euclidean
    print(f"Using {func_name} as distance function during evaluation")
    return dist_func

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
            obj = [filename, extractor(PATH_TO_DATA + '/' + filename)]
            if frame_no == 0:
                # all objects will be new
                obj.append(obj_id)
                obj_id+=1
            else:
                if ALL_FRAMES:
                    # check all the previous frames
                    max_sim = 0
                    for f in range(frame_no):
                        try:
                            if frames[f] is not None:
                                for past_frame_obj in frames[f]:
                                    sim = cos(past_frame_obj[1],obj[1])
                                    if sim >= SIMILARITY_THRESH:
                                        if sim >= max_sim:
                                            max_sim = sim
                                            obj_id_to_append = past_frame_obj[2]
                        except KeyError as e:
                            pass
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
        if obj_lst is not None:
            for obj in obj_lst:
                reid.append([frame,obj[0],obj[2]])

    reid_df = pd.DataFrame(reid, columns=['frame#', 'filename', 'id'])
    reid_df.to_csv(PATH_TO_OUTPUT + '/reid_torchreid.csv')

def centroidsReid(PATH_TO_DATA, PATH_TO_OUTPUT='.'):

    os.system('python ./centroids_reid/inference/create_embeddings.py --config_file="centroids_reid/configs/256_resnet50.yml" GPU_IDS [0] DATASETS.ROOT_DIR "'+PATH_TO_DATA+'" TEST.IMS_PER_BATCH 128 OUTPUT_DIR "output_dir" TEST.ONLY_TEST True MODEL.PRETRAIN_PATH "centroids_reid/market1501_resnet50_256_128_epoch_120.ckpt"')

    paths = np.load('output_dir/paths.npy', allow_pickle=True)

    embeddings = np.load('output_dir/embeddings.npy', allow_pickle=True)

    paths_embeddings = []
    for i in range(len(paths)):
        paths_embeddings.append([paths[i], embeddings[i]])

    paths_embeddings = np.array(paths_embeddings,dtype='object')
    # print(paths)

    paths_embeddings = np.array(sorted(paths_embeddings, key=lambda x: int(x[0][23:-4])))
    paths = paths_embeddings[:,0]
    embeddings = list(paths_embeddings[:,1])

    for i in range(len(embeddings)):
        embeddings[i] = list(embeddings[i])

    embeddings = np.array(embeddings)

    paths = paths.astype('<U')
    embeddings = embeddings.astype('float32')

    embeddings = torch.nn.functional.normalize(
                torch.from_numpy(embeddings), dim=1, p=2
            )

    # Use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    # embeddings_gallery = embeddings_gallery.to(device)
    embeddings = embeddings.to(device)

    ### Calculate similarity
    print("Calculating distance and getting the most similar ids per query")
    dist_func = get_dist_func('cosine')
    distmat = dist_func(x=embeddings, y=embeddings).cpu().numpy()
    indices = np.argsort(distmat, axis=1)

    ### Constrain the results to only topk most similar ids
    topk = 10
    indices = indices[:, : topk] if topk else indices

    out = {
        query_path: {
            "indices": indices[q_num, :],
            "paths": paths[indices[q_num, :]],
            "distances": distmat[q_num, indices[q_num, :]],
        }
        for q_num, query_path in enumerate(paths)
    }

    ### Save
    SAVE_DIR = Path('output_dir')
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Saving results to {str(SAVE_DIR)}")
    np.save(SAVE_DIR / "results.npy", out)

    results = np.load('output_dir/results.npy', allow_pickle=True)

    pair_dist_map = dict()

    for _,path in enumerate(paths):
        sub_paths = results.item().get(path).get('paths')
        # print(sub_paths)
        distances = results.item().get(path).get('distances')
        # print(path.split('/')[1])
        for i, sub_path in enumerate(sub_paths):
            # if i == 0: 
            if path.split('/')[2] == sub_path.split('/')[2]:
                continue
            if tuple(set([path.split('/')[2], sub_path.split('/')[2]])) not in pair_dist_map.keys():
                pair_dist_map[tuple(set([path.split('/')[2], sub_path.split('/')[2]]))] = distances[i]
                # print(path.split('/')[2], sub_path.split('/')[2], distances[i]) 

    # print(pair_dist_map)

    frame_id_map = dict()
    for path in paths:
        frame_id_map[path.split('/')[2]] = -1

    uid = 0
    for path in paths:
        # print(path)

        # print('\n******************************\npath ', path.split('/')[1])
        # print('path frame number ', path.split('/')[1].split('_')[0].split('frame')[1])
        path_fno = int(path.split('/')[2].split('_')[0].split('frame')[1])

        if path_fno == 0:
            # print(path.split('/')[2])
            frame_id_map[path.split('/')[2]] = uid
            uid += 1
            continue

        min_dist = 1
        min_key = 1
        min_pair_path = None
        check = False

        for key_pair in pair_dist_map.keys():
            # print(key_pair)
            # print(path.split('/')[2])
            # print(getitem(key_pair, 1))
            # print(getitem(key_pair, 0))

            # print(results.item().get(path))

            if path.split('/')[2] == getitem(key_pair, 0):
                pair_path = getitem(key_pair, 1)
                # print(pair_path)
            # try:
            elif path.split('/')[2] == getitem(key_pair, 1):
                pair_path = getitem(key_pair, 0)
                # print(pair_path)
            # except KeyError as e:
            #     pass
            else:
                continue

            # print('pair_path frame number ', pair_path.split('_')[0].split('frame')[1])
            pair_path_fno = int(pair_path.split('_')[0].split('frame')[1])

            if pair_dist_map[key_pair] < min_dist and path_fno > pair_path_fno:
                # print(pair_path)
                min_dist = pair_dist_map[key_pair]
                min_key = key_pair
                min_pair_path = pair_path
                check = True

        if check == True:
            # print(min_key, pair_dist_map[min_key], min_dist)      
            frame_id_map[path.split('/')[2]] = frame_id_map[min_pair_path]

        else:
            frame_id_map[path.split('/')[2]] = uid
            uid += 1

    print('Saving reid_centroids.csv')
    reid = pd.DataFrame(columns = ['frame#', 'filename', 'id'])
    for frame in frame_id_map.keys():
        # print(frame)
        reid = reid.append(pd.DataFrame([[frame.split('frame')[1].split('_')[0], frame, frame_id_map[frame]]], columns = ['frame#', 'filename', 'id']))

    reid.to_csv('reid_centroids.csv', index=False)