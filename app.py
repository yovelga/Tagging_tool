import os
import cv2
import streamlit as st



# Directory containing images (you can replace this with your actual directory)
current_dir = os.path.dirname(__file__)
image_dir = os.path.join(current_dir,"images")
image_files = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Initialize session state
if "current_image_index" not in st.session_state:
    st.session_state.current_image_index = 0

if image_files:
    current_image_path = image_files[st.session_state.current_image_index]
    image = cv2.imread(current_image_path)
    if image is not None:
        # Convert BGR to RGB for proper color display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # One row with 3 images: 1 large and 2 small
        row = st.columns([2,1, 1, 1,1,1])  # 2 parts for the large image, 1 part each for smaller images

        # Display the large image (spans 2 columns)
        with row[0]:
            st.image(image, caption="Original image", use_container_width =False)

        with row[1]:
            st.image(image, caption="extra bbox", use_container_width =False)

        # Display the first smaller image
        with row[2]:
            st.image(image, caption="Boundery box", use_container_width =False)

        # Display the second smaller image
        with row[3]:
            st.image(image, caption="Segmented object", use_container_width =False)

        with row[4]:
            st.image(image, caption="masked object", use_container_width=False)

    else:
        st.error(f"Failed to load image: {current_image_path}")
else:
    st.error("No images found in the specified directory.")
st.write("---")

# Navigation buttons
col1, col2, col3,col4,col5 = st.columns([3,3,5,5,5])
with col1:
    if st.button("Next"):
        st.session_state.current_image_index = min(len(image_files) - 1, st.session_state.current_image_index + 1)
    if st.button("Previous"):
        st.session_state.current_image_index = max(0, st.session_state.current_image_index - 1)

with col3:
    if st.button("Grape"):
        st.session_state.current_image_index = min(len(image_files) - 1, st.session_state.current_image_index + 1)
    if st.button("Not Grape"):
        st.session_state.current_image_index = min(len(image_files) - 1, st.session_state.current_image_index + 1)

# Separator for spacing
st.write("---")
