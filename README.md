# LK-Tracker
**LK-Tracker** is a program that uses the Lucas-Kanade tracking algorithm with pyramids to follow key edge and corner points in a scene. The underlying mechanism used by the Lucas-Kanade algorithm is optical flow, which represents the apparent motion of objects seen by the observer and is shown through displacement vectors. Pyramids are used to downscale the image, as the algorithm fails for large motion and works well for smaller motion. Below is an example of optical flow seen as a vehicle turning on a curved road. 

<p align="center"> 
<img src="https://github.com/k22jung/kl_tracker/blob/master/examples/image1.jpg">
</p>

The original code that this project is based off of can be found [here]( https://github.com/opencv/opencv_extra/blob/master/learning_opencv_v2/ch10_ex10_1.cpp). 

## Dependencies

The program was ran and created for Ubuntu 16.04
- [OpenCV 3.2.0](http://opencv.org/releases.html)

## Running

This is an Eclipse project, you may simply compile it. Run with argument `-h` for details on how to use.




