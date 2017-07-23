
/*
 * Source taken from Github as a starting point for the code:
 *
 * https://github.com/opencv/opencv_extra/blob/master/learning_opencv_v2/ch10_ex10_1.cpp
 * Example 10-1. Pyramid Lucas-Kanade optical flow code
 *
 *  Learning OpenCV 2: Computer Vision with the OpenCV Library
 *    by Gary Bradski and Adrian Kaehler
 *    Published by O'Reilly Media
 *
*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

bool breakout = 0;

const int WIN_SIZE = 5;//5//10; WIN_SIZE*4+1
const int MAX_CORNERS = 400;//500
const int SKIP_FRAMES = 2; // Optical flow will be calculated every SKIP_FRAMES frames.
const double MAX_LOST_PERCENT = 0.4;//0.8
const int MAX_LOSSES = 300;//180;//10
const int PYRAMID_LEVELS = 20;//5;
//const int WIDTH =  1920;//960;
//const int HEIGHT =  1080;//540;
const int WIDTH =  960;
const int HEIGHT =  540;
const string WINDOWNAME = "image";
const int TARGET_FRAME = 875;
//string FILENAME = "/home/kenny/Downloads/fireworks.mp4";
//string FILENAME = "/home/kenny/Downloads/oclude.mp4";
//string FILENAME ="/home/kenny/Downloads/video.mp4";
//string FILENAME ="/home/kenny/Downloads/stopsign1.mp4";
//string FILENAME ="/home/kenny/Downloads/stopsign2.mp4";
string FILENAME ="/home/kenny/Downloads/squareone_stopsign.mp4";


// Grab image from the video capture, resizing it to desired dimensions and converting to greyscale.
inline int getImage(Mat &frame, Mat &grey, VideoCapture &cap){
    cap >> frame;

    if (frame.empty()){
    	return 1;
    }

    resize(frame, frame, Size(WIDTH, HEIGHT), 0, 0, INTER_LINEAR);
    cvtColor(frame, grey, CV_BGR2GRAY, 0);

    return 0;
}

// Draws optical flow vectors on the most recent frame, counts the number of point errors
// of the LK tracker.
inline void drawFlow(vector<uchar> &features_found, int num_points, int &num_errs, Mat &frame, vector<Point2f> &cornersA, vector<Point2f> &cornersB){
	for( int j = 0; j < num_points; j++ ) {

		if( !features_found[j]){
			 num_errs++;
			 continue;
		}

		arrowedLine(frame, cornersA[j], cornersB[j], Scalar(0,255,0), 2, CV_AA,0,0.3);
		//circle(frame2, cornersB[j], 1, Scalar(0,255,0), 2, 8, 0);
	}
}

int main(int argc, char** argv) {
while(!breakout){
	    Mat frame1, frame2, frame1_grey,frame2_grey;
	    vector<Point2f> cornersA, cornersB, temp;
	    vector<uchar> features_found;
	    cv::Mat result;
	    double lost_percent;
		int num_errs = 0;
		int num_points;
		int error;

	    VideoCapture cap(FILENAME);

	    if(!cap.isOpened())
	    	return EXIT_FAILURE;

	    namedWindow(WINDOWNAME,WINDOW_NORMAL);
	    setWindowProperty(WINDOWNAME, CV_WND_PROP_FULLSCREEN,CV_WINDOW_FULLSCREEN);
	    resizeWindow(WINDOWNAME,WIDTH,HEIGHT);

	    // Grab an image for the first iteration of optical flow.

  	    int count=0;
  	    do{
  	    if (getImage(frame1, frame1_grey, cap)){
  	        	return EXIT_FAILURE;
  	        }
  	    count++;
  	    }while(count < TARGET_FRAME);

		goodFeaturesToTrack(frame1_grey, cornersB, MAX_CORNERS, 0.1, 5, noArray(), 3, true, 0.04);//https://en.wikipedia.org/wiki/Canny_edge_detector
		num_points = cornersB.size();

	    for(int i = 0;;i++)
	    {
	    	error = getImage(frame2, frame2_grey, cap);

	    	if (error){
	    		break;
	   	    }

	    	// Optical flow is recalculated every other SKIP_FRAMES frames.
	    	if (!(i%(SKIP_FRAMES-1))){
	    		cornersA = cornersB;

				cornerSubPix(frame1_grey, cornersA, Size(WIN_SIZE, WIN_SIZE), Size(-1,-1),
							 TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));

				calcOpticalFlowPyrLK(frame1_grey, frame2_grey, cornersA, cornersB, features_found, noArray(),
									 Size(WIN_SIZE*4+1,WIN_SIZE*4+1), PYRAMID_LEVELS,
									 TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ));
			}


	    	drawFlow(features_found, num_points, num_errs, frame2, cornersA, cornersB);
			imshow(WINDOWNAME,frame2);

			// If too many lost points, re-detect more corner points and display dots instead of
			// past optical flows.
			if (!(i%(SKIP_FRAMES-1))){
				if (num_points > 0){
					lost_percent = 1.0*(num_errs)/num_points;
				}else{
					lost_percent = 1;
				}

				// Resets key points if too many losses and makes the optical flow
				// simply display a dot instead for frames before the frame containing
				// the next LK tracking iteration.
				if(lost_percent > MAX_LOST_PERCENT || num_errs > MAX_LOSSES){
					goodFeaturesToTrack(frame2_grey, cornersA, MAX_CORNERS, 0.01, 5, noArray(), 3, false, 0.04);
					cornersB = cornersA;
					num_points = cornersA.size();
				}

				frame1_grey = frame2_grey.clone();
			}

			num_errs = 0;

			if(waitKey(1) == 32)
			{
				breakout = 1;
				break;
			}
	    }


	 cap.release();
}

	 destroyAllWindows();

  return 0;
}
