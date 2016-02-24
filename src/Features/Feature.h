//
//  Feature.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef Robust_Struck_Feature_h
#define Robust_Struck_Feature_h

#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>

class Feature {

public:

    virtual ~Feature(){}
    
    virtual cv::Mat prepareImage(cv::Mat* imageIn)=0;
    
    //virtual int calculateDimension()=0;
    
   
    virtual cv::Mat calculateFeature(cv::Mat& processedImage,std::vector<cv::Rect>& rects)=0;
    virtual int calculateFeatureDimension()=0;
    
    virtual std::string getInfo()=0;

    virtual void setParams(const std::unordered_map<std::string, std::string> & map) {
    }

    cv::Mat reshapeYs(std::vector<cv::Rect>& locations){
        
        cv::Mat y = cv::Mat::zeros((int)locations.size(), 5, CV_64F);
        // for every location
        for (int l=0; l<locations.size(); ++l) {
            y.at<double>(l,0) = l;
            y.at<double>(l,1) = locations[l].x;
            y.at<double>(l,2) = locations[l].y;
            y.at<double>(l,3) = locations[l].width;
            y.at<double>(l,4) = locations[l].height;
        }
        
        return y;
    }
};

#endif
