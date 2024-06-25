import cv2

def apply_blur(frame):
    # Apply Gaussian blur with kernel size (15, 15) and sigmaX 0
    blurred_frame = cv2.GaussianBlur(frame, (45, 45), 0)
    return blurred_frame

def apply_black_and_white(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Convert grayscale frame back to BGR (3-channel) for display purposes (optional)
    bw_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    return bw_frame

def open_webcam():
    # Open the default camera (usually camera 0)
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Apply blur filter
        blurred_frame = apply_black_and_white(frame)
        
        # Display the resulting frame
        cv2.imshow('Blur Filter', blurred_frame)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()