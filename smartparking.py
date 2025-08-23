import json
from shapely.geometry import Polygon
import numpy as np
import cv2
from ultralytics import YOLO
import argparse
import os

def get_parking_slots_from_json(json_file): # json_file is the path to the JSON file containing parking slot data (json_fpath)
    # This function reads a JSON file and extracts parking slot polygons.
    with open(json_file, "r") as f:
        data = json.load(f)
    parking_slots = []
    for shape in data["shapes"]:
        if shape["shape_type"] == "polygon":
            parking_slots.append(Polygon(shape["points"]))
    return parking_slots

def object_detection(img_path):
    """
    Arguments:
        img_path: (str/nd.array (opencv)) - Path to the input image / or an OpenCV image array.
    Returns:
        car_polygons: (list) - List of detected car polygons as shapely Polygon objects.
        img: (nd.array) - Image with detected objects drawn on it.
        orig_img: (nd.array) - Original image before any modifications.
    """
    # This function performs object detection on the input image using the specified model.
    results = model(img_path)
    result = results[0]  # Get the first result
    car_polygons = []
    if not result.obb:
        print("No Oriented Bounding Boxes (OBB) detected.")
        return car_polygons, None, None
    else:
        print("Detected objects with Oriented Bounding Boxes (OBB).")
        CLASSES = result.names
        # Create random colors for each class
        colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        orig_img = result.orig_img.copy()
        img = orig_img.copy()
        xyxyxyxy = result.obb.xyxyxyxy
        class_ids = result.obb.cls
        confs = result.obb.conf
        for idx in range(len(class_ids)):
            conf = confs[idx]
            bbox = xyxyxyxy[idx].cpu().numpy() # bbox coordinates in the format [x1, y1, x2, y2, x3, y3, x4, y4]
            pts = [(int(x), int(y)) for x, y in bbox] # [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            car_polygons.append(Polygon(pts))
            label = f"{CLASSES[int(class_ids[idx])]} ({conf:.2f})" # "car (0.95)"   
            cv2.polylines(img, [np.array(pts)], True, colors[int(class_ids[idx])], 2)
            cv2.putText(img, label, (pts[0][0] - 10, pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[int(class_ids[idx])], 2)
    return car_polygons, img, orig_img

def occupied_parking_slots(parking_slots, car_polygons, threshold=0.25):
    """
    Arguments:
        parking_slots: (list) - List of parking slot polygons as shapely Polygon objects.
        car_polygons: (list) - List of detected car polygons as shapely Polygon objects.
    Returns:
        occupied: (list) - List indicating whether each parking slot is occupied.
    """
    occupied = [False] * len(parking_slots)
    for i, slot in enumerate(parking_slots):
        for car in car_polygons:
            if slot.intersects(car):
                iou = slot.intersection(car).area / (slot.area + 0.0001)  # Adding a small value to avoid division by zero
                print(f"Slot {i+1} intersects with car, IoU: {iou:.2f}")
                # Check if the intersection area is greater than a threshold
                if iou > threshold:  # Threshold for considering a slot occupied
                    occupied[i] = True
                    break
    return occupied # [True, False, True, ...] - List indicating whether each parking slot is occupied

def draw_parking_slots(img, parking_slots, occupied):
    """
    Arguments:
        img: (nd.array) - Image with detected objects drawn on it.
        parking_slots: (list) - List of parking slot polygons as shapely Polygon objects.
        occupied: (list) - List indicating whether each parking slot is occupied.
    Returns:
        img: (nd.array) - Image with parking slots drawn on it. 
    """
    for i, slot in enumerate(parking_slots):
        pts = np.array(slot.exterior.coords).astype(np.int32)
        color = (0, 0, 255) if occupied[i] else (0, 255, 0)  # Red for occupied, Green for available
        cv2.polylines(img, [pts], True, color, 3)
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(img, f"Slot {i+1}", (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Parking Slot Detection")
    parser.add_argument("--model_path", type=str, default="best.pt", help="Path to the YOLO model file.")
    parser.add_argument("--json_fpath", type=str, default="layout.json", help="Path to the JSON file containing parking slot data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image folder, single image file, or video file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file (for video) or folder (for images).")
    parser.add_argument("--threshold", type=float, default=0.25, help="Threshold for considering a parking slot occupied based on IoU.")
    args = parser.parse_args()

    # Make sure output directory exists only if input is a folder or multiple images
    if not args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and not os.path.isfile(args.input): 
        os.makedirs(args.output, exist_ok=True)

    # Load parking slots from JSON file
    parking_slots = get_parking_slots_from_json(args.json_fpath)

    # Load the YOLO model one time
    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} does not exist.")
        exit(1)
    else:
        print(f"Loading model from {args.model_path}...")
        model = YOLO(args.model_path)
    # Perform object detection on random input image before processing the actual image
    im = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)  # Random image for testing
    out = model(im)  # Perform inference on the random image

    # Check if input is a video file
    is_video = args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        # Process video
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.input}")
            exit(1)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            print(f"Processing frame {frame_count}")

            # Perform detection on the frame
            car_polygons, processed_frame, _ = object_detection(frame)
            
            if car_polygons:
                # Check occupied parking slots
                occupied = occupied_parking_slots(parking_slots, car_polygons, threshold=args.threshold)
                # Draw parking slots on the frame
                processed_frame = draw_parking_slots(processed_frame, parking_slots, occupied)
            
            # Write the frame
            out.write(processed_frame)

        # Release everything
        cap.release()
        out.release()
        print(f"Video processing completed. Output saved to {args.output}")

    else:
        # Process image(s)
        img_paths = []
        if os.path.isfile(args.input):
            img_paths = [args.input]
        else:
            # List all images in the input folder
            images = [x for x in os.listdir(args.input) if x.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            img_paths = [os.path.join(args.input, f) for f in images]

        if not img_paths:
            print("No images found in the specified input path.")
            exit(1)

        # Create output directory if processing multiple images
        if len(img_paths) > 1:
            os.makedirs(args.output, exist_ok=True)

        for img_path in img_paths:
            print(f"Processing image: {img_path}")
            car_polygons, img, _ = object_detection(img_path)
            
            if car_polygons:
                print(f"Detected {len(car_polygons)} cars in the image.")
                occupied = occupied_parking_slots(parking_slots, car_polygons, threshold=args.threshold)
                img = draw_parking_slots(img, parking_slots, occupied)
            else:
                print("No cars detected in the image.")
                img = cv2.imread(img_path)

            # Save the output
            if len(img_paths) > 1:
                output_path = os.path.join(args.output, os.path.basename(img_path))
            else:
                output_path = args.output
            
            cv2.imwrite(output_path, img)
            print(f"Saved processed image to {output_path}")

# Usage examples:
# For video: python smartparking.py --model_path yolo_model.pt --json_fpath parking_slots.json --input input_video.mp4 --output output_video.mp4
# For single image: python smartparking.py --model_path yolo_model.pt --json_fpath parking_slots.json --input input.jpg --output output.jpg
# For image folder: python smartparking.py --model_path yolo_model.pt --json_fpath parking_slots.json --input input_folder --output output_folder
