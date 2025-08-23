import cv2
import os
import argparse
from pathlib import Path
import numpy as np

def images_to_video(image_folder, output_path, fps=30, sort_by_name=True, max_frames=None):
    """
    Convert a folder of images into a video file.
    
    Args:
        image_folder (str): Path to the folder containing images
        output_path (str): Path where the output video will be saved
        fps (int): Frames per second for the output video
        sort_by_name (bool): Whether to sort images by filename
        max_frames (int, optional): Maximum number of frames to include in the video

    Returns:
        bool: True if successful, False otherwise
    """
    # Get list of image files
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in valid_extensions:
        image_files.extend(list(Path(image_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(image_folder).glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return False
    
    # Sort images if requested
    if sort_by_name:
        image_files.sort()
    
    # Read the first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"Error reading image: {image_files[0]}")
        return False
    
    height, width = first_image.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each image
    total_images = len(image_files)
    for i, image_path in enumerate(image_files):
        if max_frames is not None and i >= max_frames:
            break
        print(f"Processing image {i+1}/{total_images}: {image_path.name}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
            
        # Ensure image has the same dimensions as the first image
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        
        # Write frame
        out.write(image)
    
    # Release video writer
    out.release()
    print(f"Video saved to {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a folder of images to video")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing images")
    parser.add_argument("--output", type=str, required=True, help="Output video file path")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video")
    parser.add_argument("--no-sort", action="store_false", dest="sort", 
                      help="Don't sort images by filename")
    parser.add_argument("--max-frames", type=int, default=None, 
                      help="Maximum number of frames in the output video")
    
    args = parser.parse_args()
    
    # Ensure input folder exists
    if not os.path.isdir(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        exit(1)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert images to video
    success = images_to_video(args.input, args.output, args.fps, args.sort, args.max_frames)
    if not success:
        exit(1)
