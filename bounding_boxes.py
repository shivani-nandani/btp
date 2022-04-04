import cv2
import pandas as pd

def drawBoundingBoxes(imageData, imageOutputPath, inferenceResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,r,b)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    for res in inferenceResults:
        left = int(res['left'])
        top = int(res['top'])
        right = int(res['right'])
        bottom = int(res['bottom'])
        label = res['label']
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        # print(left, top, right, bottom)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
        cv2.putText(imageData, label, (left, top - 12), 0, 1e-3 * imgHeight, color, thick)
    cv2.imwrite(imageOutputPath, imageData)


def getBoundingBoxes(reid_results_path, detection_results_path, frames_path):
    """Get bounding boxes

    Args:
        reid_results_path (string): path to reid results csv 
        detection_results_path (string): path to detection results csv
        frames_path (string): path to frames directory 
    """


    reid_results = pd.read_csv(reid_results_path)
    detection_results = pd.read_csv(detection_results_path)

    for i in range(len(reid_results)):

        imgcv = cv2.imread(frames_path+'/'+reid_results.filename[i].split('_')[0]+'.jpg')
        color = (255,0,0)
        results = [
            {
                'left' : detection_results.xmin[int(reid_results.filename[i].split('_')[1].split('.')[0])],
                'top' : detection_results.ymin[int(reid_results.filename[i].split('_')[1].split('.')[0])],
                'right' : detection_results.xmax[int(reid_results.filename[i].split('_')[1].split('.')[0])],
                'bottom' : detection_results.ymax[int(reid_results.filename[i].split('_')[1].split('.')[0])],
                'label': 'P'+str(reid_results.id[i])
            }
        ]
        drawBoundingBoxes(imgcv, frames_path+'/'+reid_results.filename[i].split('_')[0]+'.jpg', results, color)