import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load a pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Preprocess the image
def preprocess(image_path):
    input_image = Image.open(image_path).convert('RGB')
    original_size = input_image.size  # Store the original size (width, height)
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch, np.array(input_image), original_size

# Get the segmentation mask
def get_segmentation_mask(input_batch):
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions.byte().cpu().numpy()

# Resize the mask to the original image size
def resize_mask(mask, original_size):
    mask_image = Image.fromarray(mask)
    resized_mask = mask_image.resize(original_size, Image.NEAREST)
    return np.array(resized_mask)

# Apply the mask to the image
def apply_mask(image, mask):
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 15] = [255, 255, 255]  # Person class in DeepLabV3
    return cv2.bitwise_and(image, colored_mask)

def segment_image(path):
    # Load and preprocess the image
    image_path = path #'IMG_0034.png'
    input_batch, original_image, original_size = preprocess(image_path)

    # Get the segmentation mask
    mask = get_segmentation_mask(input_batch)

    # Resize the mask to the original image size
    resized_mask = resize_mask(mask, original_size)

    # Apply the resized mask to the original image
    segmented_image = apply_mask(original_image, resized_mask)

    image_warmer = segmented_image.copy().astype(np.float32)

    # Increase red and yellow tones (R, G, B channels)
    image_warmer[:, :, 2] *= 1.2  # Scale red channel
    image_warmer[:, :, 1] *= 1.1  # Scale green channel slightly
    image_warmer[:, :, 0] *= 0.9  # Scale blue channel slightly

    # Ensure pixel values are within valid range [0, 255]
    image_warmer = np.clip(image_warmer, 0, 255).astype(np.uint8)

    # Save the segmented image
    segmented_image_path = 'segmented_IMG_0034.png'
    cv2.imwrite(segmented_image_path, cv2.cvtColor(image_warmer, cv2.COLOR_RGB2BGR))

    # Display the original and segmented images
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.imshow(original_image)
    # plt.title('Original Image')
    # plt.subplot(122)
    # plt.imshow(segmented_image)
    # plt.title('Segmented Image')
    # plt.show()

    print(f'Segmented image saved as {segmented_image_path}')
    return segmented_image_path