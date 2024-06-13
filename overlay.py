import cv2
import numpy as np

def overlay(path1, path2):
    output_path = 'final_blurred_output.png'

    # Load images
    image1 = cv2.imread(path1) # blurred background
    image2 = cv2.imread(path2) # segmented image
 
    # Check original dimensions
    print(f"Image 1 dimensions: {image1.shape[:2]}")
    print(f"Image 2 dimensions: {image2.shape[:2]}")

    # Resize images to match the dimensions of image1 (assuming image1 is the base)
    width, height = image1.shape[1], image1.shape[0]
    image2_resized = cv2.resize(image2, (width, height))

    # Overlay image2 onto image1
    blended_image = np.copy(image1)
    blended_image[:, :, :] = np.where(image2_resized > 0, image2_resized, blended_image)

    # Display the overlaid image
    # cv2.imshow('Overlaid Image', blended_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the overlaid image
    cv2.imwrite(output_path, blended_image)
    return output_path