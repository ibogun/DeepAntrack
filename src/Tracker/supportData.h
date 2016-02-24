//
//  supportData.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 12/15/14.
//
//

#ifndef __Robust_tracking_by_detection__supportData__
#define __Robust_tracking_by_detection__supportData__

#include <stdio.h>
#include <opencv2/opencv.hpp>

class supportData {
public:
    cv::Mat x;
    cv::Mat y;
    cv::Mat beta;
    int label;
    cv::Mat grad;
    int frameNumber;
    supportData();

    supportData(const cv::Mat& x_,const cv::Mat& y_,const int& label_, const int&m, const int& K, int frameNumber_){


        x= cv::Mat(x_);
        y= cv::Mat(y_);
        beta=cv::Mat::zeros(1,K, CV_64F);
        grad= cv::Mat::zeros(1,K, CV_64F);
        label=label_;
        frameNumber=frameNumber_;
    }
    ~supportData(){
    }
};

#endif /* defined(__Robust_tracking_by_detection__supportData__) */
