//
//  DrawRandomImage.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 2/9/15.
//
//

#ifndef __Robust_tracking_by_detection__DrawRandomImage__
#define __Robust_tracking_by_detection__DrawRandomImage__

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

class DrawRandomImage {

public:
    /// Global Variables
    const int NUMBER = 100;
    const int DELAY = 0;

    int seed;
    
    const int window_width = 3;
    const int window_height = 2;
    int x_1 = -window_width/2;
    int x_2 = window_width*3/2;
    int y_1 = -window_width/2;
    int y_2 = window_width*3/2;
    

    /// Function headers
    DrawRandomImage() {this->seed = 1;
        srand(this->seed);};
    DrawRandomImage(int seed_) {this->seed = seed_;
        srand(this->seed);};
    static cv::Scalar randomColor(cv::RNG& rng );
    int Drawing_Random_Lines( cv::Mat image, char* window_name, cv::RNG rng );
    int Drawing_Random_Rectangles( cv::Mat image, char* window_name, cv::RNG rng );
    int Drawing_Random_Ellipses( cv::Mat image, char* window_name, cv::RNG rng );
    int Drawing_Random_Polylines( cv::Mat image, char* window_name, cv::RNG rng );
    int Drawing_Random_Filled_Polygons( cv::Mat image, char* window_name, cv::RNG rng );
    int Drawing_Random_Circles( cv::Mat image, char* window_name, cv::RNG rng );
    int Displaying_Random_Text( cv::Mat image, char* window_name, cv::RNG rng );
    int Displaying_Big_End( cv::Mat image, char* window_name, cv::RNG rng );
    
    
    cv::Mat getRandomImage();
    cv::Mat getRandomFloatMatrix();

    cv::Mat getRandomBoundingBox(int rows, int cols);
};

#endif /* defined(__Robust_tracking_by_detection__DrawRandomImage__) */
