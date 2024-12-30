import streamlit as st
from PIL import Image
import cv2
import numpy as np
from backhand import (
    load_jsons,
    load_current_json,
    process_detection,
    next_image,
    prev_image,
    next_detection,
    prev_detection,
    current_json_index,
    current_detection_index,
    json_files,
    detections_len,
    update_current_json_tag,
    next_detection_index,
)

# Initialize JSONs
load_jsons()


if not json_files:
    st.rerun()
else:
    # Load the current JSON
    data = load_current_json()

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("Next Image"):
            next_image()
        if st.button("Previous Image"):
            prev_image()
        if st.button("Next Detection"):
            next_detection()
        if st.button("Previous Detection"):
            prev_detection()
        if st.button('Refresh Page'):
            st.rerun ()

    # Process current detection

    st.write("---")

    # Display current detection details
    current_detection = data[current_detection_index]
    results = process_detection(current_detection)
    st.write(f"Current Tag: {current_detection['tag']}")
    st.write(f"JSON File: {json_files[current_json_index]} (Index {current_json_index + 1}/{len(json_files)})")
    st.write(f"Detection {current_detection_index + 1}/{detections_len}")



    # Display images
    row = st.columns([3, 1, 1, 1, 1, 1])

    st.subheader("Annotated Images")
    with row[0]:
        st.image(results["annotated_image"], caption="Annotated Full Image", use_container_width=True)
    with row[1]:
        st.image(results["cropped_extended_image"], caption="Cropped Extended Image", use_container_width=True)
    with row[2]:
        st.image(results["annotated_cropped_extended_image"], caption="Annotated Cropped Extended Image", use_container_width=True)
    with row[3]:
        st.image(results["masked_image_cropped_extended"], caption="Masked Cropped Extended Image", use_container_width=True)

    # Navigation Buttons
    with row[4]:
        if st.button("Next Untagged Image"):
            # Navigate to the next untagged image
            index = current_detection_index + 1
            found = False

            while not found:
                # Check if the current detection is untagged
                if data[index]["tag"] == "none":
                    next_detection_index(index)
                    next_detection()
                    found = True
                    st.rerun()
                    break

                # Move to the next detection
                index += 1

                # Check if we've reached the end of the current file
                if index >= detections_len:
                    next_image()
                    data = load_current_json()
                    index = 0
                    detections_len = len(data)

            if found:
                st.success(f"Moved to the next untagged image at index {current_detection_index + 1}.")
            else:
                st.warning("No untagged images found.")

        if st.button("Next"):
            next_detection()
            st.rerun()
        if st.button("Previous"):
            prev_detection()
            st.rerun()
        if st.button("Grape"):
            update_current_json_tag("Grape")
            next_detection()
            st.rerun()
        if st.button("Not Grape"):
            update_current_json_tag("Not_grape")
            next_detection()
            st.rerun()
