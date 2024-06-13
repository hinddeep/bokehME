import cv2

def blur_background(path):
    output_path = 'background_only_blurred.png'

    # Read the image
    image_path = path
    image = cv2.imread(image_path)

    # Apply Gaussian blur with a larger kernel and higher standard deviation
    blurred_image = cv2.GaussianBlur(image, (35, 35), 25)

    # Save the blurred image
    cv2.imwrite(output_path, blurred_image)

    # Display the original and blurred images
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Blurred Image', blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return output_path