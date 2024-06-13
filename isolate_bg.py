import cv2
import torch
import numpy as np
from torchvision import transforms, models

def remove_subject(path):
    output_path = 'background_only.png'

    # Load DeepLabv3+ pretrained model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Define preprocessing transforms
    preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((513, 513)),  # Resize to match model's expected size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the input image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, convert to RGB

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)  # Get the index of the maximum value

    # Create a mask where 1 = background, 0 = foreground (subject)
    background_mask = np.where(output_predictions.numpy() == 0, 255, 0).astype(np.uint8)

    # Resize mask to match original image size
    background_mask = cv2.resize(background_mask, (image.shape[1], image.shape[0]))

    # Apply the mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=background_mask)

    # Convert result back to BGR for OpenCV
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    # Save the result image
    cv2.imwrite(output_path, result_image)

    print("Background-only image saved as background_only.png")
    return output_path