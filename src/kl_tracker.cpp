/* Kenneth Jung
 * Mechatronics Engineering
 * k22jung@edu.uwaterloo.ca
 *
 * Original source taken from Github as a starting point for the code:
 *
 * https://github.com/opencv/opencv_extra/blob/master/learning_opencv_v2/ch10_ex10_1.cpp
 * Example 10-1. Pyramid Lucas-Kanade optical flow code
 *
 *  Learning OpenCV 2: Computer Vision with the OpenCV Library
 *    by Gary Bradski and Adrian Kaehler
 *    Published by O'Reilly Media
*/

#include <opencv2/opencv.hpp>
#include "opencv2/core/utility.hpp"
#include <iostream>

using namespace cv;
using namespace std;

bool breakout = false;

int win_size = 5;
int max_corners = 400;
int skip_frames = 2; // Optical flow will be calculated every SKIP_FRAMES frames.
const double MAX_LOST_PERCENT = 0.4;
const int MAX_LOSSES = 300;
int pyramid_levels = 5;
int width =  960;
int height =  540;
const string WINDOWNAME = "Video";
int target_frame = 0;


const String keys =
    "{help h usage ? |                  | print this message                                                }"
    "{@video         |../squareone_stopsign.mp4 | Path to video file.                                       }"
	"{win-size | 5 | Window size for optical flow computation.}"
	"{max-corners | 400  | Maximum number of corner and edge points detected to track in a scene.}"
	"{frame-skip | 2 | Skip every other frame-skip frames for optical flow computation (loss in accuracy, but faster runtime). }"
	"{pyr-levels | 5 | Assign number of pyramid levels for optical flow. }"
	"{width |960  | Set display window width.}"
	"{height |540  | Set display window height.}"
	"{ff | 0 | Fast-forward to this number frame in the video.}"
	;


// Grab image from the video capture, resizing it to desired dimensions and converting to greyscale.
inline int getImage(Mat &frame, Mat &grey, VideoCapture &cap){
    cap >> frame;

    if (frame.empty())
    {
    	return 1;
    }

    resize(frame, frame, Size(width, height), 0, 0, INTER_LINEAR);
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

int main(int argc, char** argv)
{
	int key;
	CommandLineParser parser(argc,argv,keys);
	parser.about("Lucas-Kanade Scene Tracking");
	if (parser.has("help"))
	{
		parser.printMessage();
	    return 0;
	}

	String filename = parser.get<String>(0);
    win_size = parser.get<int>("win-size");
    max_corners = parser.get<int>("max-corners");
    skip_frames = parser.get<int>("frame-skip");
    pyramid_levels = parser.get<int>("pyr-levels");
    width = parser.get<int>("width");
    height = parser.get<int>("height");
    target_frame = parser.get<int>("ff");

	namedWindow(WINDOWNAME,WINDOW_NORMAL);
	setWindowProperty(WINDOWNAME, CV_WND_PROP_FULLSCREEN,CV_WINDOW_FULLSCREEN);
	resizeWindow(WINDOWNAME,width,height);

	while(!breakout)
	{
		Mat frame1, frame2, frame1_grey,frame2_grey;
		vector<Point2f> cornersA, cornersB, temp;
		vector<uchar> features_found;
		cv::Mat result;
		double lost_percent;
		int num_errs = 0;
		int num_points=0;
		int error;

		VideoCapture cap(filename);

		if(!cap.isOpened())
		{
			cout << "Can't open video capture." << endl;
			return EXIT_FAILURE;
		}

		// Grab an image for the first iteration of optical flow.
		int count = 0;
		do{
			if (getImage(frame1, frame1_grey, cap))
			{
				return EXIT_FAILURE;
			}
			count++;
		}while(count < target_frame);

		goodFeaturesToTrack(frame1_grey, cornersB, max_corners, 0.1, 5, noArray(), 3, true, 0.04);

		for(int i = 0;;i++)
		{
			error = getImage(frame2, frame2_grey, cap);

			if (error)
			{
				break;
			}

			// Optical flow is recalculated every other SKIP_FRAMES frames.
			if (!(i%(skip_frames-1)))
			{
				cornersA = cornersB;
				num_points = cornersA.size();

				cornerSubPix(frame1_grey, cornersA, Size(win_size, win_size), Size(-1,-1),
							 TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));

				calcOpticalFlowPyrLK(frame1_grey, frame2_grey, cornersA, cornersB, features_found, noArray(),
									 Size(win_size*4+1,win_size*4+1), pyramid_levels,
									 TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ));
			}


			drawFlow(features_found, num_points, num_errs, frame2, cornersA, cornersB);
			imshow(WINDOWNAME,frame2);

			// If too many lost points, re-detect more corner points and display dots instead of
			// past optical flows.
			if (!(i%(skip_frames-1)))
			{
				if (num_points > 0)
				{
					lost_percent = 1.0*(num_errs)/num_points;
				}
				else
				{
					lost_percent = 1;
				}

				// Resets key points if too many losses and makes the optical flow
				// simply display a dot instead for frames before the frame containing
				// the next LK tracking iteration.
				if(lost_percent > MAX_LOST_PERCENT || num_errs > MAX_LOSSES)
				{
					goodFeaturesToTrack(frame2_grey, cornersA, max_corners, 0.01, 5, noArray(), 3, false, 0.04);
					cornersB = cornersA;
				}

				frame1_grey = frame2_grey.clone();
			}

			num_errs = 0;

			key = waitKey(1);
			if(key == ' ' || key == 'q' || key == 'Q')
			{
				breakout = true;
				break;
			}
		}

		cap.release();
	}

	destroyAllWindows();

  return 0;
}
