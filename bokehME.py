from isolate_subject import segment_image
from isolate_bg import remove_subject
from blur_bg import blur_background
from overlay import overlay

from separate_video import separate_sub_bg
from process_video import process_sub_bg
from overlay_video import overlay_video

from webcam_blur import open_webcam

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Apply filters to the sunject and background of images and video streams in realtime')
    parser.add_argument('-f', '--file', type=str, required=False, help='Path of local image/video to be processed')
    parser.add_argument('-b', '--blur', action='store_true', help='Blur the source image/video')
    parser.add_argument('-i', '--image', action='store_true', help='Process and image')
    parser.add_argument('-v', '--video', action='store_true', help='Process a video')
    parser.add_argument('-w', '--webcam', action='store_true', help='Open Webcam and apply filters')
    args = parser.parse_args()

    if args.image and args.blur:
            # Step 1: Isolate subject from the background
            segmented_image_path = segment_image(args.file)

            # Step 2: Remove subject from the original image
            background_only_path = remove_subject(args.file)

            # Step 3: Blur the background
            blurred_background_path = blur_background(background_only_path)

            # Step 4: Overlay the subject on the blurred background
            overlay(blurred_background_path, segmented_image_path)

            # Step 5: Cleanup, delete all intermediate outputs
            try:
                # Attempt to delete the file
                os.remove(segmented_image_path)
                os.remove(background_only_path)
                os.remove(blurred_background_path)
            except OSError as e:
                print(f"Error deleting the file: {e.filename} - {e.strerror}")

    if args.video and args.blur:
        videos = separate_sub_bg(args.file)
        process_sub_bg()
        overlay_video('blurred_video.mp4', 'processed_subject.mp4')

        try:
                # Attempt to delete the file
                os.remove(videos[0])
                os.remove(videos[1])
                os.remove('blurred_video.mp4')
        except OSError as e:
                print(f"Error deleting the file: {e.filename} - {e.strerror}")

    if args.webcam:
        open_webcam()

if __name__ == '__main__':
    main()