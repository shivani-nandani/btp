from frames_videos import videoToFrames, framesToVideo
from object_detection import objectDetection, cropPersons
from reid import torchreidReid
from bounding_boxes import getBoundingBoxes


# create frames from the original video 
path_to_frames = videoToFrames(input_path='street.mp4')
# print('Frames saved at '+path_to_frames)

# detect objects in the frames 
objectDetection(model_version='yolov5x', frames_input_path=path_to_frames, detection_results_path='detection_results.csv', total_frames=330)

# generating video from detected frames 
framesToVideo(fps=30, output_path='detected.avi')

# crop individuals 
cropPersons(detection_results_path='detection_results.csv', image_directory_path='./frames', cropped_image_output_path='./cropped_persons', confidence_threshold=0.5)

# re-identification
torchreidReid(SIMILARITY_THRESH=0.8, PATH_TO_DATA='./cropped_persons', PATH_TO_OUTPUT='.')

# create bounding boxes for each reid 
getBoundingBoxes(reid_results_path='reid.csv', detection_results_path='detection_results.csv', frames_path='./frames')

# generating video from reid frames 
framesToVideo(fps=30, input_path='./frames', output_path='reid.avi')
