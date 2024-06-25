import threading
import time
import cv2
import numpy as np

# Define two functions that will run on separate threads
def process_subject():
    # Step 1: Open the video file using cv2.VideoCapture
    video_path = 'subject_video.mp4'  # Replace with your video file path
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Step 2: Get video properties (width, height, frames per second)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Step 3: Create video writer object to save processed video
    out_path = 'processed_subject.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Step 4: Read and process each frame
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            break  # Break the loop if the video is over

        # Step 5: Apply warmer effect to the frame
        warmer_frame = frame.astype(np.float32)
        warmer_frame[:, :, 2] *= 1.2  # Increase red channel
        warmer_frame[:, :, 1] *= 1.1  # Increase green channel slightly
        warmer_frame[:, :, 0] *= 0.9  # Decrease blue channel slightly

        # # Ensure pixel values are within valid range [0, 255]
        warmer_frame = np.clip(warmer_frame, 0, 255).astype(np.uint8)

        # Step 6: Write the processed frame to the output video file
        out.write(warmer_frame)

        # Check for 'q' key pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Step 7: Release the video capture and writer objects, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_background():
    input_file = 'background_video.mp4'
    output_file = 'blurred_video.mp4'  # Output video file
    new_background = cv2.imread('new_bg.png')
    blur_strength = 15

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
        
        background_resized = cv2.resize(new_background, (frame.shape[1], frame.shape[0]))

        diff = cv2.absdiff(frame, background_resized)
        # Threshold the difference image
        _, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)

        # Invert the mask
        mask_inv = np.uint8(cv2.bitwise_not(mask))

        # Ensure mask_inv has the same size as image2
        mask_inv = cv2.resize(mask_inv, (background_resized.shape[1], background_resized.shape[0]))

        # Apply the mask to image2 to remove overlapping parts
        image2_removed = cv2.bitwise_and(background_resized, background_resized, mask=mask_inv)

        # Convert frame to Metal-compatible format
        frame_metal = cv2.UMat(image2_removed)

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

def process_sub_bg():
    # Create threads for each function
    thread1 = threading.Thread(target=process_subject)
    thread2 = threading.Thread(target=process_background)

    # Start both threads
    thread1.start()
    thread2.start()

    print("Both threads started...")
    # Wait for both threads to finish
    thread1.join()
    thread2.join()

    # Both threads have finished
    print("Both threads have finished, continuing with the main program")