import streamlit as st
import cv2
import pickle
import numpy as np
import cvzone
import tempfile
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image
# Page Configuration
st.set_page_config(page_title="AI Parking Space Detector", layout="wide")
st.title("🅿️ Real-Time Parking Space Detector")
st.markdown("This application uses Computer Vision to detect available parking spots from video feeds.")

# 1. Setup Sidebar for Parameters
st.sidebar.header("Settings")
pixel_threshold = st.sidebar.slider("Sensitivity (Pixel Count)", 500, 2000, 900, help="Lower values make detection stricter.")
width, height = 120, 80

# 2. Load Coordinates (The Annotations)
# Create a 'data' folder if it doesn't exist
# --- PATHS ---
DATA_PATH = './data/position_list2'
DIM_PATH = './data/dimension_list'
IMAGE_PATH = './data/frame_to_annotate.jpg'


if not os.path.exists('data'):
    os.makedirs('data')



def check_parking_space(img, img_process):
    space_counter = 0
    for pos, dim in zip(posList,dim_list):
        x, y = pos
        width,height = dim
        img_crop = img_process[y:y+height, x:x+width]
        count = cv2.countNonZero(img_crop)

        # Logic for occupancy
        if count < pixel_threshold:
            color = (0, 255, 0) # Green
            thickness = 3
            space_counter += 1
        else:
            color = (0, 0, 255) # Red
            thickness = 2

        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 5), scale=1, 
                           thickness=1, offset=0, colorR=color)

    # Big Header for results
    cvzone.putTextRect(img, f'Free: {space_counter}/{len(posList)}', (40, 50), 
                       scale=3, thickness=3, offset=20, colorR=(0, 200, 0))
    return img

# 3. File Uploader
video_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Save video to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # Placeholder for the video stream
    frame_window = st.image([])
    # Check if positions exist
    if not os.path.exists(DATA_PATH):
        st.warning("No parking spots detected. We need to annotate a frame first!")
        
        # --- STEP 2: CAPTURE FRAME ---
        cap = cv2.VideoCapture(tfile.name)
        success, frame = cap.read()
        if success:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bg_image = Image.fromarray(frame_rgb)
            
            st.subheader("Draw boxes over the parking spots")
            st.info("Select the 'Rect' tool on the left and draw your spots.")

            # 2. Create the Interactive Canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Orange with transparency
                stroke_width=2,
                background_image=bg_image,
                update_streamlit=True,
                height=bg_image.height,
                width=bg_image.width,
                drawing_mode="rect",
                key="canvas",
            )
            # 3. Process the results when the user is done
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if objects:
                    if st.button("Save Selected Spots"):
                        new_pos_list = []
                        dimenison_list= []
                        for obj in objects:
                            # Get the top-left (x, y) of the drawn rectangle
                            x, y = int(obj['left']), int(obj['top'])
                            w, h = int(obj['width']), int(obj['height'])
                            new_pos_list.append((x, y))
                            dimenison_list.append((w,h))

                        
                        # Save to your pickle file
                        with open(DATA_PATH, 'wb') as f:
                            pickle.dump(new_pos_list, f)
                        with open(DIM_PATH, 'wb') as f:
                            pickle.dump(dimenison_list, f)                        
                        st.success(f"Saved {len(new_pos_list)} spots! You can now run the detection.")
                        # Provide a button to jump to the main app
                        if st.button("Go to Detection Mode"):
                            st.session_state.mode = "detect"
                            st.rerun()
        st.stop()
    
    # --- STEP 3: DETECTION MODE ---
    else:
        with open(DATA_PATH, 'rb') as f:
            posList = pickle.load(f)
        with open(DIM_PATH,'rb') as f:
            dim_list = pickle.load(f)            
        st.success(f"Loaded {len(posList)} parking spots. Starting detection...")
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            # Loop video for the demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Image Processing Pipeline
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
        img_threshold = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 25, 16)
        img_median = cv2.medianBlur(img_threshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        img_dilate = cv2.dilate(img_median, kernel, iterations=1)

        # Detection logic
        processed_img = check_parking_space(img, img_dilate)

        # Convert BGR (OpenCV) to RGB (Streamlit)
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        
        # Display in Streamlit
        frame_window.image(processed_img_rgb)
else:
    st.info("Please upload a video file in the sidebar to start the detection.")