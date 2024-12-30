import streamlit as st
from PIL import Image
import cv2
import numpy as np
from back import (
    load_tif_files,
    get_current_tif,
    process_tif_detection,
    next_tif,
    prev_tif,
    save_metadata_to_tif
)

# Load TIF files
tif_files = load_tif_files()

if not tif_files:
    st.error("No TIF files found.")
    st.stop()

# Load the current TIF file
current_tif_data = get_current_tif()
tif_path = current_tif_data["path"]
metadata = current_tif_data["metadata"]

if not metadata or "image_name" not in metadata:
    st.error("Metadata or image name is missing in the TIF file.")
    st.stop()

# Process current detection
results = process_tif_detection(tif_path)

if not results or "annotated_image" not in results:
    st.error("Failed to process the TIF file.")
    st.stop()

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

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    if st.button("Next TIF"):
        next_tif()
        st.rerun()
    if st.button("Previous TIF"):
        prev_tif()
        st.rerun()
    if st.button("Next Untagged TIF"):
        next_untagged_tif()
        st.rerun()
    if st.button("Refresh Page"):
        st.rerun()

# Display TIF file details
st.write("---")
st.write(f"Current TIF: {tif_path}")
st.write(f"Metadata: {metadata}")

# Display images
row = st.columns([3, 1, 1, 1])

st.subheader("Processed Images")
with row[0]:
    st.image(results["annotated_image"], caption="Annotated Image", use_container_width=True)
with row[1]:
    st.image(results["cropped_image"], caption="Cropped Image", use_container_width=True)
with row[2]:
    st.image(results["cropped_extended_image"], caption="Cropped Extended Image", use_container_width=True)
with row[3]:
    st.image(results["masked_image_cropped"], caption="Masked Cropped Image", use_container_width=True)

# Tagging Options
st.write("---")
st.subheader("Tagging")

if st.button("Tag as Grape"):
    metadata["tag"] = "Grape"  # Update the tag
    save_metadata_to_tif(tif_path, metadata)  # Save the updated metadata back to the TIF
    st.success("Tagged as Grape and saved.")
    next_tif()  # Move to the next TIF file
    st.rerun()

if st.button("Tag as Not Grape"):
    metadata["tag"] = "Not Grape"  # Update the tag
    save_metadata_to_tif(tif_path, metadata)  # Save the updated metadata back to the TIF
    st.success("Tagged as Not Grape and saved.")
    next_tif()  # Move to the next TIF file
    st.rerun()



