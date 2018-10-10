///////////////////////////////////////////////////////////////////////////////////
/// GestureDetection_Rpi.cpp
///
/// Detects hand gestures and controls GPIO
/// Dependencies: 
/// OpenCV, WiringPi, C++11
///
///////////////////////////////////////////////////////////////////////////////////

// Check if RPI, else just do detection
#ifdef __arm__
#include <wiringPi.h>
#endif

#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;

// function declarations
void DetectGesture(cv::Mat frame);
void sendData(int n);
void setSerial();
bool markHand(cv::Mat frame, vector<Point> cont,  Point *rPoint,int *radius);
vector<Point> mark_fingers(Mat frame, vector<Point> hull,Point pt,int radius, int *fingerCnt);

// global vars  
int USB = open( "/dev/ttyUSB0", O_RDWR | O_NOCTTY | O_NONBLOCK);
bool first_iteration=true;
float finger_ct_history[] = {0.0,0.0};

int main()
{

    cv::Mat img;

    cv::VideoCapture capture(0);
    capture.set(CAP_PROP_FPS, 30);
    setSerial();

    {
        while (capture.isOpened())
        {
            capture >> img;

            if (!img.empty())
            {
                
		DetectGesture(img);
                //cv::imshow("live",img);
                cv::waitKey(1);
            }
            
            // If escape key is pressed, bail out
            if (cv::waitKey(100) == 27)
                break;
        }
    }
    return 0;
}

void DetectGesture(cv::Mat frame)
{
    cv::Mat img, binary;
     frame.convertTo(frame,-1,0.3,-10);
    cvtColor(frame, img, COLOR_BGR2GRAY);	//black and white


    //thresholding  
    cv::threshold(img, binary, 35, 255, cv::THRESH_BINARY_INV); 

    //Contour storage
    vector<vector<Point>> Contours;
    vector<Vec4i> hierarchy;    

    //morphological transformations
	
    erode(binary, binary, getStructuringElement(MORPH_RECT, Size(4, 4))); 
    erode(binary, binary, getStructuringElement(MORPH_RECT, Size(4, 4)));    
    dilate(binary, binary, getStructuringElement(MORPH_RECT, Size(9, 9)));

    cv::imshow("SegmentedOut", binary);
    
    //finding the contours required
    findContours(binary, Contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    //finding the contour of largest area and storing its index
    int biggestContour = 0;
    int maxarea = 0;

    //convex Hull storage
    vector<Point> cnvxHull;

    for (int i = 0; i < Contours.size(); i++)
    {
        double currentArea = contourArea(Contours[i]);
        if (currentArea > maxarea)
        {
            maxarea = currentArea;
            biggestContour = i;
        }
    }    
    convexHull(Contours[biggestContour], cnvxHull, false);

    Point handPos;	
    int handRadius;
    int fingerCnt=0;
    bool handScore = markHand(frame,Contours[biggestContour],&handPos,&handRadius);
    vector<Point> fingers = mark_fingers(frame,cnvxHull,handPos,handRadius,&fingerCnt);

    if (maxarea > 100)
    {
	if (handScore) {
		circle(frame,handPos,handRadius,Scalar(255,0,0),4);
		for (int k=0;k<fingers.size();k++)
		{
			circle(frame, Point(fingers[k].x,fingers[k].y), 13, Scalar(255, 255, 0), 4);
		}
	}
    }
    
    printf("Fingers: %d\n",fingerCnt);
    sendData(fingerCnt);
	
    cv::imshow("Live", frame);
}


void sendData(int nFingers)
{

     	char message = nFingers;
	char response[100];
	
	write( USB, &message, 1 );

	int i=0;
	int cnt;
	char buf;
	do
	{
	  
	   cnt = read( USB, &buf, 1 );
	   if (cnt < 1)
		response[i]='\0';
	   else
		response[i] = buf;
	   i++;
	   
	} while(buf != '\n' && cnt > 0);
	
	    
	printf("Response: %s\n",response);
}

void setSerial()
{
	struct termios tty;
	struct termios tty_old;
	memset (&tty, 0, sizeof tty);
	
	/* Error Handling */
	if ( tcgetattr ( USB, &tty ) != 0 ) {
	   std::cout << "Error " << errno << " from tcgetattr: " << strerror(errno) << std::endl;
	}
	
	/* Save old tty parameters */
	tty_old = tty;
	
	/* Set Baud Rate */
	cfsetospeed (&tty, (speed_t)B115200);
	cfsetispeed (&tty, (speed_t)B115200);
	
	/* Setting other Port Stuff */
	tty.c_cflag     &=  ~PARENB;            // Make 8n1
	tty.c_cflag     &=  ~CSTOPB;
	tty.c_cflag     &=  ~CSIZE;
	tty.c_cflag     |=  CS8;
	
	tty.c_cflag     &=  ~CRTSCTS;           // no flow control
	tty.c_cc[VMIN]   =  1;                  // read doesn't block
	tty.c_cc[VTIME]  =  0.5;                  // 0.5 seconds read timeout
	tty.c_cflag     |=  CREAD | CLOCAL;     // turn on READ & ignore ctrl lines
	
	/* Make raw */
	cfmakeraw(&tty);
	
	/* Flush Port, then applies attributes */
	tcflush( USB, TCIFLUSH );
	if ( tcsetattr ( USB, TCSANOW, &tty ) != 0) {
	   std::cout << "Error " << errno << " from tcsetattr" << std::endl;
	}
}



bool markHand(cv::Mat frame, vector<Point> cont,  Point *rPoint,int *radius)
{
    int max_d=0;
    Point pt(0,0);
    Rect rect = boundingRect(cont);
    int x = rect.x;
    int y = rect.y;
    int w = rect.width;
    int h = rect.height;

    int ind_y;
    int ind_x;

    for (ind_y=int(y+0.3*h); ind_y <= int(y+0.8*h); ind_y+=5)
        for( ind_x=int(x+0.3*w); ind_x <= int(x+0.6*w) ; ind_x+=5) {
            
		
	    int dist= pointPolygonTest(cont,Point2f(ind_x,ind_y),true);
            if(dist>max_d) {

                max_d=dist;
                pt.x=ind_x;
		pt.y=ind_y;
	    }
	}
    
    
     *radius = max_d;
     rPoint->x = pt.x;
     rPoint->y = pt.y;
     
     if(max_d > 0.04*frame.cols)
        return true;
     else
        return false;

    
}

vector<Point> mark_fingers(Mat frame, vector<Point> hull,Point pt,int radius, int *fingerCnt)
{
    vector<Point> finger;
    int finger_count = 0;

    finger.push_back( Point(hull[0].x,hull[0].y) );
    int j=0;
    int i=0;

    int cx = (int)(pt.x);
    int cy = (int)(pt.y);
    
    for(i=0;i<hull.size()-1;i++) {
        float dist = sqrt( (hull[i].x - hull[i+1].x)*(hull[i].x - hull[i+1].x) + (hull[i].y - hull[i+1].y)*(hull[i].y - hull[i+1].y));
        
		if (dist>18) {
            if(j==0) {
                finger.clear();
				finger.push_back( Point(hull[i].x,hull[i].y) );
			}
            else
                finger.push_back( Point(hull[i].x,hull[i].y) );
            j=j+1;
		}
    }

    
    int temp_len=finger.size();
    i=0;
    while(i<temp_len) {
        float dist = sqrt( (finger[i].x - cx)*(finger[i].x - cx) + (finger[i].y - cy)*(finger[i].y - cy) ) ;
        
	if(dist<2*radius || dist>3.8*radius || finger[i].y>cy+radius) {
            finger.erase(finger.begin()+i);
            temp_len=temp_len-1;
	}
        else
            i=i+1 ; 
    }   


    
    temp_len=finger.size();
    if(temp_len>5)
        for (i=1;i<=temp_len+1-5;i++)
            finger.erase(finger.begin()+temp_len-i);
    
    
  

    if(first_iteration) {
        finger_ct_history[0]=finger_ct_history[1]=finger.size();
        first_iteration=false;
    }
    else
        finger_ct_history[0]=0.34*(finger_ct_history[0]+finger_ct_history[1]+finger.size());

    if( (finger_ct_history[0]-int(finger_ct_history[0])) > 0.8)
        finger_count=int(finger_ct_history[0])+1;
    else
        finger_count=int(finger_ct_history[0]);

    finger_ct_history[1]=finger.size();
    *fingerCnt = finger_count;
    

   return finger;
}

