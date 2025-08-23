from ultralytics import YOLO
import numpy as np
import cv2
import ipdb

def draw_bounding_box(img, class_id, confidence, x1y1, x2y2, x3y3, x4y4):
    """
    Draw bounding boxes on the input image based on the provided arguments.

    Args:
        img (np.ndarray): The input image to draw the bounding box on.
        class_id (int): Class ID of the detected object.
        confidence (float): Confidence score of the detected object.
        x1 (int): X-coordinate of the top-left corner of the bounding box.
        y1 (int): Y-coordinate of the top-left corner of the bounding box.
        x2 (int): X-coordinate of the bottom-right corner of the bounding box.
        y2 (int): Y-coordinate of the bottom-right corner of the bounding box.
    """
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    points = np.array([x1y1, x2y2, x3y3, x4y4])
    cv2.polylines(img, [points], True, color, 2)
    cv2.putText(img, label, (x1y1[0] - 10, x1y1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    

# Load a model
model = YOLO("best.pt")  # load a pretrained model (recommended for training)

# Predict with the model
results = model("car_parking.jpg")  # predict on an image
# results[0].show()

# Access the results
result = results[0]
CLASSES = result.names
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# ipdb.set_trace()
orig_img = result.orig_img
img = orig_img.copy()
xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()  # polygon format with 4-points
class_ids = result.obb.cls.cpu().numpy().astype(np.int32)
confs = result.obb.conf  # confidence score of each box
for idx, class_id in enumerate(class_ids):
    conf = confs[idx]
    bbox = xyxyxyxy[idx, :]
    x1y1 = (int(bbox[0, 0]), int(bbox[0, 1]))
    x2y2 = (int(bbox[1, 0]), int(bbox[1, 1]))
    x3y3 = (int(bbox[2, 0]), int(bbox[2, 1]))
    x4y4 = (int(bbox[3, 0]), int(bbox[3, 1]))
    # points = bbox.astype(np.int32)
    draw_bounding_box(img, class_id, conf, x1y1, x2y2, x3y3, x4y4)
print(result.obb)  # Will show None if OBBs aren't available


cv2.imwrite("./output.jpg", img)
    


    