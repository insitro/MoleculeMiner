import time

import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import detection_predictor


def nms(bounding_boxes: np.ndarray, confidence_score: np.ndarray, threshold: float):
    """
    Non-Maximum Suppression Algorithm

    Parameters
    ----------
    bounding_boxes : np.ndarray
        Bounding boxes of detected text
    confidence_score : np.ndarray
        Confidence scores of bounding boxes
    threshold : float
        Threshold for intersection-over-union

    Returns
    -------
    np.ndarray
        Picked bounding boxes
    np.ndarray
        Picked confidence scores
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return np.array(picked_boxes).reshape((-1, 4)), picked_score


def detect_text():
    eg_file = (
        "/home/ec2-user/projects/chem_data_extraction/table-transformer/"
        + "data/test_pdfs_imgs/US20230660011A1_Voroni_477_molecules/Page_00051_cell0.png"
    )
    # model = ocr_predictor(pretrained=True)
    model = detection_predictor(arch="db_resnet50", pretrained=True)
    doc = DocumentFile.from_images(eg_file)
    result = model(doc)
    meta = result[0]["words"]
    # boxes = boxes[boxes[:,4] > 0.31]
    boxes = meta[:, :4]
    scores = meta[:, 4]
    boxes, _ = nms(boxes, scores, threshold=0.98)
    # Normalize by width and height
    img = cv2.imread(eg_file)
    height, width = img.shape[:2]
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height

    # Visualize the boxes
    for box in boxes:
        start = [int(box[0]), int(box[1])]
        end = [int(box[2]), int(box[3])]
        img = cv2.rectangle(img, start, end, (0, 0, 255), thickness=2)

    cv2.imwrite("char_detections.png", img)
    # pages = result.synthesize()
    # plt.axis('off')
    # plt.imsave('Test_TableOCR.png', pages[0].astype(np.uint8))


if __name__ == "__main__":
    start = time.time()
    detect_text()
    print(f"Total time: {time.time() - start}")
