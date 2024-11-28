import json
import os
import cv2
import numpy as np


import base64

# Directory containing JSON files
json_dir = r"C:\Users\yovelg\Desktop\VS_projects\git_projects\tagging_tool\ObjectJSONizer\JSONs"

# Global variables for navigation
json_files = []
current_json_index = 0
current_detection_index = 0
detections_len = 0


def load_jsons():
    """Populate the list of JSON files."""
    global json_files
    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(json_dir)
        for file in files if file.endswith('.json')
    ]
    if not json_files:
        print("No JSON files found.")
        raise FileNotFoundError("No JSON files found in the directory.")
    # print(f"Found JSON Files: {json_files}")



def load_json_data(json_file_path):
    """Load data from a JSON file."""
    try:
        with open(json_file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {json_file_path}: {e}")
        return []


def decode_mask(mask_base64):
    """Decode a base64-encoded mask."""
    try:
        mask_bytes = base64.b64decode(mask_base64)
        mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_np, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError("Decoded mask is None.")
        print(f"Decoded mask shape: {mask.shape}")  # Debugging line
        return mask
    except Exception as e:
        print(f"Error decoding mask: {e}")
        return None



def process_detection(detection):
    """Process a single detection."""
    try:
        results = process_images(
            detection["image_path"],
            decode_mask(detection["mask"]),
            detection["original_bbox"],
            detection["expanded_bbox"],
        )
        print("Processed Results:", results)  # Debugging output
        return results
    except Exception as e:
        print(f"Error in process_detection: {e}")
        return {}

def process_images(image_path, mask, bbox, extended_bbox):
    try:
        # Function to load and validate the image
        def load_and_check_image(image_path):
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")
            if len(image.shape) < 2 or image.size == 0:
                raise ValueError(f"Invalid or empty image at {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"Image loaded successfully: {image_path}")
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            return image

        # Load and validate the image
        image = load_and_check_image(image_path)

        # Convert bounding box coordinates to integers
        bbox = list(map(int, bbox))
        extended_bbox = list(map(int, extended_bbox))

        # Ensure bounding boxes are within image bounds
        height, width, _ = image.shape
        bbox = [
            max(0, min(width, bbox[0])),
            max(0, min(height, bbox[1])),
            max(0, min(width, bbox[2])),
            max(0, min(height, bbox[3]))
        ]
        extended_bbox = [
            max(0, min(width, extended_bbox[0])),
            max(0, min(height, extended_bbox[1])),
            max(0, min(width, extended_bbox[2])),
            max(0, min(height, extended_bbox[3]))
        ]

        # Crop the image and mask
        image_cropped = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        image_cropped_extended = image[extended_bbox[1]:extended_bbox[3], extended_bbox[0]:extended_bbox[2]]

        if mask.shape[:2] != image.shape[:2]:
            raise ValueError("Mask dimensions do not match image dimensions")

        mask_cropped = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        mask_cropped_extended = mask[extended_bbox[1]:extended_bbox[3], extended_bbox[0]:extended_bbox[2]]

        # Ensure masks are binary
        mask = (mask > 0).astype(np.uint8)
        mask_cropped = (mask_cropped > 0).astype(np.uint8)
        mask_cropped_extended = (mask_cropped_extended > 0).astype(np.uint8)

        # Function to create a purple mask
        def create_purple_mask(mask):
            """
            Converts a binary mask into a purple RGB mask.

            Parameters:
                mask (numpy.ndarray): A binary mask (0 or 1) with the same size as the image.

            Returns:
                numpy.ndarray: A purple mask in RGB format.
            """
            if len(mask.shape) != 2:
                raise ValueError("Input mask must be a 2D array.")

            # Create the purple mask by setting red (R) and blue (B) channels to 255
            # and the green (G) channel to 0.
            purple_mask = cv2.merge([
                mask * 255,  # Blue channel (B)
                mask * 0,    # Green channel (G)
                mask * 255   # Red channel (R)
            ])

            return purple_mask

        # Annotate images by overlaying masks
        annotated_image = cv2.addWeighted(image.copy(), 0.7, create_purple_mask(mask), 0.3, 0)
        annotated_cropped_image = cv2.addWeighted(image_cropped, 0.7, create_purple_mask(mask_cropped), 0.3, 0)
        annotated_cropped_extended_image = cv2.addWeighted(image_cropped_extended, 0.7, create_purple_mask(mask_cropped_extended), 0.3, 0)

        # Return processed images
        return {
            "image": image,
            "cropped_image": image_cropped,
            "cropped_extended_image": image_cropped_extended,
            "annotated_image": annotated_image,
            "masked_image_cropped": cv2.bitwise_and(image_cropped, image_cropped, mask=mask_cropped),
            "masked_image_cropped_extended": cv2.bitwise_and(image_cropped_extended, image_cropped_extended, mask=mask_cropped_extended),
            "annotated_cropped_image": annotated_cropped_image,
            "annotated_cropped_extended_image": annotated_cropped_extended_image,
        }

    except Exception as e:
        print(f"Error in process_images: {e}")
        return {
            "error": str(e),
            "annotated_image": None,
            "masked_image_cropped": None,
            "masked_image_cropped_extended": None,
            "annotated_cropped_image": None,
            "annotated_cropped_extended_image": None,
        }


def next_detection():
    """Move to the next detection."""
    global current_detection_index, detections_len

    if detections_len > 0:
        # Increment the detection index
        current_detection_index += 1

        # If the index exceeds available detections, load the next image
        if current_detection_index >= detections_len:
            current_detection_index = 0  # Reset to the first detection in the next file
            next_image()  # Move to the next image



def next_detection_index(new_index):
    """Move to the next detection."""
    global current_detection_index, detections_len
    current_detection_index = new_index -1

def prev_detection():
    """Move to the previous detection."""
    global current_detection_index

    if current_detection_index > 0:
        current_detection_index -= 1
    else:
        # Move to the last detection of the previous image
        prev_image()
        current_json_path = json_files[current_json_index]
        data = load_json_data(current_json_path)
        current_detection_index = len(data) - 1 if data else 0



def next_image():
    """Move to the next image."""
    global current_json_index, current_detection_index, detections_len

    if current_json_index < len(json_files) - 1:
        current_json_index += 1
        data = load_current_json()
        current_detection_index = 0  # Start from the first detection
        detections_len = len(data)
    else:
        print("Reached the last image. No more files to load.")



def prev_image():
    """Move to the previous image."""
    global current_json_index, current_detection_index, detections_len

    if current_json_index > 0:
        current_json_index -= 1
        data = load_current_json()
        current_detection_index = len(data) - 1 if data else 0  # Start from the last detection
        detections_len = len(data)
    else:
        print("Reached the first image. No previous files to load.")



def load_current_json():
    """Load the current JSON file and update detection length."""
    global detections_len
    current_json_path = json_files[current_json_index]
    data = load_json_data(current_json_path)
    detections_len = len(data)
    return data

def update_current_json_tag(tag):
    """
    Updates the `tag` of the currently viewed object in the current JSON file.

    Parameters:
        tag (str): The new tag to assign to the current detection.

    Returns:
        bool: True if the tag was updated and JSON saved successfully, False otherwise.
    """
    global json_files, current_json_index, current_detection_index

    try:
        # Load the current JSON file
        if not json_files:
            print("No JSON files loaded.")
            return False

        current_json_path = json_files[current_json_index]
        json_data = load_json_data(current_json_path)

        # Validate index and JSON structure
        if not isinstance(json_data, list):
            raise ValueError("JSON data must be a list of detections.")
        if current_detection_index < 0 or current_detection_index >= len(json_data):
            raise IndexError("Current detection index is out of range.")

        # Update the tag of the current detection
        json_data[current_detection_index]["tag"] = tag
        print(f"Updated tag for detection {current_detection_index} in {current_json_path} to '{tag}'.")

        # Save the updated JSON back to the file
        return save_updated_json(current_json_path, json_data)
    except Exception as e:
        print(f"Error updating JSON tag: {e}")
        return False

def save_updated_json(json_file_path, json_data):
    """
    Saves the updated JSON data back to the file.

    Parameters:
        json_file_path (str): The path to the JSON file.
        json_data (list): The updated JSON data.

    Returns:
        bool: True if the JSON file was saved successfully, False otherwise.
    """
    try:
        with open(json_file_path, 'w') as file:
            json.dump(json_data, file, indent=4)
        print(f"Saved updated JSON to {json_file_path}.")
        return True
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return False


def get_current_detection():
    """Get the current detection safely."""
    global current_detection_index, detections_len
    if 0 <= current_detection_index < detections_len:
        current_json_path = json_files[current_json_index]
        data = load_json_data(current_json_path)
        return data[current_detection_index]
    return None


