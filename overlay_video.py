import cv2

def overlay_video(video1_path, video2_path):
    output_path = 'output_video.mp4'  # Path to output video

    # Open videos
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # Get properties
    width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video2.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret1 or not ret2:
            break

        # Resize frame1 to match frame2 dimensions
        frame1 = cv2.resize(frame1, (width, height))

        # Overlay frame1 onto frame2
        blended = cv2.addWeighted(frame2, 1, frame1, 0.5, 0)

        # Write the frame
        out.write(blended)

        # cv2.imshow('Overlay', blended)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release everything
    video1.release()
    video2.release()
    out.release()
    cv2.destroyAllWindows()