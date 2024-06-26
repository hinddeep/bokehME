# bokehME
This code can be used to separate and process the subject and background of any image or video stream in realtime using multithreading. 

## Quick Start

All you need to do is supply the path of the source image to be blurred as a command line argument.

```
usage: bokehME.py [-h] [-f FILE] [-b] [-i] [-v] [-w]

Apply filters to the sunject and background of images and video streams in realtime

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Path of local image/video to be processed
  -b, --blur            Blur the source image/video
  -i, --image           Process an image
  -v, --video           Process a video
  -w, --webcam          Open Webcam and apply filters

eg. Usage:
python bokehME.py -b -i -f <image_src.extension>
python bokehME.py -b -v -f <video_src.extension>
python bokehME.py -w
```

## Algorithm
- Isolate the subject and background of the supplied image using ML techniques and save them. Use PyTorch and DeepLab’s pre-trained models to segment the image / video / webcam stream. 
- Spawn 2 threads and process subject on one and background on the other 
- Apply Gaussian blur filter or any other filter to the image / video / webcam stream
- Wait until both the threads return
- Superimpose/overlay the processed subject on the processed background
- Save the combination as a new image or video / output the webcam stream

## Key Challenges 
- Powerful editors such as those offered by Apple allow users to adjust the background blur of a portrait photo / video. However, sometimes we may find ourselves in a situation where we didn’t take a portrait photo / video but later wished to have the background blurred.
- One may resort to popular image processing libraries like OpenCV. However, standard filters such as Gaussian blur cannot distinguish between the subject and the background. They apply blur to the entire photo / video frame by frame
- No existing tool offers the flexibility to compose complex effect. This code allows you to not only build compound effects by combining multiple basic effects but also apply them to subject and background in parallel 
- Finally, applications such as Powerpoint and WhatsApp do not natively support editing of a live video stream from a webcam. OBS virtual cam be used to pass the processed output of this python code to a virtual camera and can be used with such apps to add support for realtime webcam feed processing - for free! 

## Reference
For further reference please read my blog:
A noob’s guide to adding stunning bokeh effects to the backgrounds of ‘non-portrait photos’
<br>
[bokehME](https://medium.com/@hinddeep.purohit007/a-noobs-guide-to-adding-stunning-bokeh-effects-to-the-backgrounds-of-non-portrait-photos-41dee873a80a)

```
https://medium.com/@hinddeep.purohit007/a-noobs-guide-to-adding-stunning-bokeh-effects-to-the-backgrounds-of-non-portrait-photos-41dee873a80a
```

[AI, a friend or a foe?](https://medium.com/@hinddeep.purohit007/a-noobs-guide-to-adding-stunning-bokeh-effects-to-the-backgrounds-of-non-portrait-photos-41dee873a80a)
```
https://medium.com/@hinddeep.purohit007/ai-a-friend-or-a-foe-e7e555d2d61e 
```
