# CUDA-ORB
ORB (Oriented FAST and Rotated BRIEF) keypoints detection, descriptors computation and BF-matching by CUDA

# Requirement
OpenCV, CUDA

# Result
Keypoints example:
![orb_show1](https://user-images.githubusercontent.com/46698134/212448682-3b7a76b9-3d9d-4a11-980e-bd3cf11d8b5c.jpg)

Matching example:
![orb_show_matched](https://user-images.githubusercontent.com/46698134/212448688-b6ccc011-f638-4936-8691-af370c920781.jpg)

# Speed
Repeat the matching test for 100 times took an average of 3.7ms on NVIDIA GeForce GTX 1080:
![timecost](https://user-images.githubusercontent.com/46698134/212448728-d4fa359c-5fff-487a-b4f7-9f29e892327f.png)

# Reference
https://github.com/opencv/opencv
http://www.gwylab.com/download/ORB_2012.pdf
