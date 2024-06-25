import cv2
import numpy as np

def blur_video_metal(input_file, blur_strength=25):
    output_file = 'blurred_video.mp4'  # Output video file
    # Open the video file
    video_capture = cv2.VideoCapture(input_file)

    # Get the video frame dimensions
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30.0, (frame_width, frame_height))

    # Create Metal context
    cv2.setUseOptimized(True)  # Enable optimized mode
    cv2.setNumThreads(8)  # Adjust number of threads as needed

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert frame to Metal-compatible format
        frame_metal = cv2.UMat(frame)

        # Apply blur effect using Metal backend
        blurred_frame_metal = cv2.GaussianBlur(frame_metal, (blur_strength, blur_strength), 0)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(blurred_frame_metal, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR) # need to convert single channel image back to BGR for further processing 

        # Convert blurred frame back to numpy array
        blurred_frame = gray_frame.get()

        # Write the blurred frame to the output video file
        out.write(blurred_frame)

    # Release video capture and writer
    video_capture.release()
    out.release()

    return output_file