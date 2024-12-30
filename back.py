import os
import json
import cv2
import numpy as np
import tifffile as tiff

# Directory containing TIF files
tif_dir = r"C:\Users\yovelg\Desktop\VS_projects\git_projects\tagging_tool\ObjectJSONizer\masks"
image_dir = r"C:\Users\yovelg\Desktop\VS_projects\git_projects\tagging_tool\streamlit_tagging\images"

# Global variables for navigation
tif_files = []
current_tif_index = 0

def load_tif_files():
    """Populate the list of TIF files."""
    global tif_files
    tif_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(tif_dir)
        for file in files if file.endswith('.tif')
    ]
    if not tif_files:
        raise FileNotFoundError("No TIF files found in the directory.")
    return tif_files

def load_tif_metadata(tif_path):
    """Load metadata from TIF file's description."""
    try:
        with tiff.TiffFile(tif_path) as tif:
            # Access tags and metadata
            if not tif.pages:
                raise ValueError("No pages found in the TIF file.")

            tags = tif.pages[0].tags
            description = tags.get("ImageDescription")  # Read the image description

            if description and description.value:
                # Convert the description to a JSON dictionary
                metadata = json.loads(description.value)
                print(f"Loaded metadata: {metadata}")
                return metadata
            else:
                raise ValueError(f"No metadata found in {tif_path}.")
    except Exception as e:
        print(f"Error loading metadata from {tif_path}: {e}")
        return {}

def decode_mask(tif_path):
    """Load the mask from the TIF file."""
    try:
        with tiff.TiffFile(tif_path) as tif:
            return tif.pages[0].asarray()
    except Exception as e:
        print(f"Error loading mask from {tif_path}: {e}")
        return None

def process_tif_detection(tif_path):
    """Process a single TIF file detection."""
    try:
        metadata = load_tif_metadata(tif_path)
        mask = decode_mask(tif_path)

        image_name = metadata.get("image_name")
        bbox = metadata.get("original_bbox", [0, 0, 100, 100])
        padded_bbox = metadata.get("padded_bbox", [0, 0, 120, 120])

        image_path = os.path.join(image_dir, image_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return process_images(image, mask, bbox, padded_bbox)
    except Exception as e:
        print(f"Error in process_tif_detection: {e}")
        return {}

def process_images(image, mask, bbox, extended_bbox):
    try:
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
        return {}

def next_tif():
    """Move to the next TIF file."""
    global current_tif_index
    if current_tif_index < len(tif_files) - 1:
        current_tif_index += 1
    else:
        print("Reached the last TIF file.")

def prev_tif():
    """Move to the previous TIF file."""
    global current_tif_index
    if current_tif_index > 0:
        current_tif_index -= 1
    else:
        print("Reached the first TIF file.")

def get_current_tif():
    """Get the current TIF file and its metadata."""
    global current_tif_index
    if not tif_files:
        print("No TIF files loaded.")
        return None

    tif_path = tif_files[current_tif_index]
    metadata = load_tif_metadata(tif_path)
    return {
        "path": tif_path,
        "metadata": metadata
    }


def save_metadata_to_tif(tif_path, metadata):
    """Save updated metadata back to the TIF file."""
    try:
        with tiff.TiffFile(tif_path, mode='r+b') as tif:
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)

            # Update the ImageDescription tag
            with tiff.TiffWriter(tif_path, append=True) as writer:
                writer.write(tif.pages[0].asarray(), description=metadata_json)

        print(f"Metadata saved successfully to {tif_path}")
    except Exception as e:
        print(f"Error saving metadata to {tif_path}: {e}")

# Function to jump to the next untagged TIF file
def next_untagged_tif():
    global tif_files
    global current_tif_data

    for i in range(len(tif_files)):
        next_tif()  # Move to the next TIF
        current_tif_data = get_current_tif()
        if current_tif_data and current_tif_data["metadata"].get("tag") == "none":
            return  # Found the next untagged TIF
    st.warning("No untagged TIF files found.")