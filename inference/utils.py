# inference/utils.py

import os
import cv2
import requests
from pathlib import Path

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video file and save them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.
        frame_rate (int): Save one frame every `frame_rate` frames.

    Returns:
        list: List of paths to the saved frame images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Failed to open video file at {video_path}")

    print("Extracting frames...")
    frame_count = 0
    saved_frame_paths = []

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        # Save one frame every `frame_rate` frames
        if frame_count % frame_rate == 0:
            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frame_paths.append(frame_path)
            print(f"Saved frame: {frame_path}")

        frame_count += 1

    video_capture.release()
    print(f"Extraction complete. Saved {len(saved_frame_paths)} frames.")
    return saved_frame_paths


def run_roboflow_inference(image_paths, api_url, api_key):
    """
    Send images to the Roboflow API and retrieve predictions.

    Args:
        image_paths (list): List of paths to the images to process.
        api_url (str): Roboflow API endpoint (e.g., https://detect.roboflow.com/<model-id>).
        api_key (str): Your Roboflow API key.

    Returns:
        dict: A dictionary mapping image paths to their Roboflow predictions.
    """
    predictions = {}

    for image_path in image_paths:
        # Ensure the image file exists
        if not Path(image_path).exists():
            print(f"Image file not found: {image_path}")
            continue

        # Read the image file as bytes
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        # Prepare the API request
        full_api_url = f"{api_url}?api_key={api_key}"
        files = {'file': image_bytes}
        response = requests.post(full_api_url, files=files)

        # Process the API response
        if response.status_code == 200:
            predictions[image_path] = response.json()
            print(f"Processed {image_path}: {response.json()}")
        else:
            print(f"Failed to process {image_path}: {response.text}")
            predictions[image_path] = None

    return predictions