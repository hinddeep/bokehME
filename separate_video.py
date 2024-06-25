import torch
import torchvision
import cv2
import numpy as np
from torchvision import models
from torchvision.transforms import transforms as T

# Function to preprocess the image
def preprocess(image, device):
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((520, 520)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0).to(device)

# Function to postprocess the output
def postprocess(output, orig_shape):
    output = output.squeeze().cpu().detach().numpy()
    output = output.argmax(0)
    output = cv2.resize(output, orig_shape[::-1], interpolation=cv2.INTER_NEAREST)
    return output

def separate_sub_bg(path):
    background_video = 'background_video.mp4'
    subject_video = 'subject_video.mp4'

    # Check if MPS (Metal Performance Shaders) backend is available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Load the pre-trained DeepLabV3 model and move it to the selected device
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()

    # Path to the local video file
    video_path = path

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_subject = cv2.VideoWriter(subject_video, fourcc, fps, (frame_width, frame_height))
    out_background = cv2.VideoWriter(background_video, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_shape = frame.shape[:2]
        input_tensor = preprocess(frame, device)
        with torch.no_grad():
            output = model(input_tensor)['out']

        # Get the segmentation mask
        seg_mask = postprocess(output, orig_shape)

        # Create a binary mask for the subject (person class is usually index 15 in COCO dataset)
        binary_mask = (seg_mask == 15).astype(np.uint8) * 255

        # Extract the subject using the binary mask
        subject = cv2.bitwise_and(frame, frame, mask=binary_mask)
        
        # warmer_subject = subject.copy().astype(np.float32)
        # warmer_subject[:, :, 2] *= 1.2  # Increase red channel
        # warmer_subject[:, :, 1] *= 1.1  # Increase green channel slightly
        # warmer_subject[:, :, 0] *= 0.9  # Decrease blue channel slightly

        # # Ensure pixel values are within valid range [0, 255]
        # warmer_subject = np.clip(warmer_subject, 0, 255).astype(np.uint8)

        # Create the background by inverting the binary mask
        background_mask = cv2.bitwise_not(binary_mask)
        background = cv2.bitwise_and(frame, frame, mask=background_mask)

        # Write the frames into the files
        out_subject.write(subject)
        out_background.write(background)

    # Release everything when job is finished
    cap.release()
    out_subject.release()
    out_background.release()

    return subject_video, background_video