from isolate_subject import segment_image
from isolate_bg import remove_subject
from blur_bg import blur_background
from overlay import overlay
import os
import sys

# Check if arguments are provided
if len(sys.argv) > 1:
    # sys.argv[0] is the script name itself
    # sys.argv[1:] are the arguments passed
    src_path = sys.argv[1]
else:
    print("No arguments provided.")

# Step 1: Isolate subject from the background
segmented_image_path = segment_image(src_path)

# Step 2: Remove subject from the original image
background_only_path = remove_subject(src_path)

# Step 3: Blur the background
blurred_background_path = blur_background(background_only_path)

# Step 4: Overlay the subject on the blurred background
final_blurred_output_ = overlay(blurred_background_path, segmented_image_path)

# Step 5: Cleanup, delete all intermediate outputs
try:
    # Attempt to delete the file
    os.remove(segmented_image_path)
    os.remove(background_only_path)
    os.remove(blurred_background_path)
except OSError as e:
    print(f"Error deleting the file: {e.filename} - {e.strerror}")