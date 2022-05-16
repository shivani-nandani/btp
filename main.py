import cv2

from frames_videos import videoToFrames, framesToVideo
from object_detection import objectDetection, cropPersons
from reid import torchreidReid, centroidsReid
from bounding_boxes import getBoundingBoxes


video_path = './street.mp4'
conf_thresh = 0.2

video = cv2.VideoCapture(video_path)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
method = 0                                          # 0 for torchreid, 1 for centroids-reid 

# create frames from the original video
print('Converting to frames...')

path_to_frames = videoToFrames(input_path=video_path)
print('Frames saved at '+path_to_frames)

# detect objects in the frames
print('Detecting objects...')

objectDetection(model_version='yolov5x', frames_input_path=path_to_frames, detection_results_path='detection_results.csv', total_frames=length)

# generating video from detected frames
print('Creating detection results video...')

framesToVideo(fps=fps, output_path='detected.avi')

# crop individuals
print('Creating persons images...')

cropPersons(detection_results_path='detection_results.csv', image_directory_path='./frames', cropped_image_output_path='./cropped_persons', confidence_threshold=conf_thresh)

# re-identification
print('Re-identification task...')

if method == 0:
    torchreidReid(SIMILARITY_THRESH=0.8, PATH_TO_DATA='./cropped_persons', PATH_TO_OUTPUT='.')
elif method == 1:
    centroidsReid(PATH_TO_DATA='./cropped_persons', PATH_TO_OUTPUT='.')

# create bounding boxes for each reid
print('Creating bounding boxes...')

if method == 0:
    getBoundingBoxes(reid_results_path='reid_torchreid.csv', detection_results_path='detection_results.csv', frames_path='./frames')
elif method == 1:
    getBoundingBoxes(reid_results_path='reid_centroids.csv', detection_results_path='detection_results.csv', frames_path='./frames')

# generating video from reid frames
print('Creating reid video...')

if method == 0:
    framesToVideo(fps=fps, input_path='./frames', output_path='reid_torchreid.avi')
elif method == 1:
    framesToVideo(fps=fps, input_path='./frames', output_path='reid_centroids.avi')

