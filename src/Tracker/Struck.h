//
//  Struck.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/6/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __Robust_Struck__Struck__
#define __Robust_Struck__Struck__

#include <stdio.h>
#include <unordered_map>
#include <vector>

#include "OLaRank_old.h"
#include "LocationSampler.h"

#include "../Features/Feature.h"

#include "../Kernels/AllKernels.h"

#include "../Features/AllFeatures.h"

#include "../Datasets/Dataset.h"
#include "../Superpixels/Plot.h"


class Struck {
public:
    OLaRank_old* olarank;
    Feature* feature;
    LocationSampler* samplerForUpdate;
    LocationSampler* samplerForSearch;
    Plot* objPlot;

    cv::Rect lastLocation;
    cv::Rect lastLocationFilter;
    cv::Rect lastLocationObjectness;
    cv::Rect lastRectFilterAndDetectorAgreedOn;
    cv::Rect gtBox;

    std::unordered_map<int, cv::Mat> frames;

    // for plotting support vectors (used only if display==2)
    cv::Mat canvas;

    cv::Mat objectnessCanvas;

    const static int gridSearch = 2;
    const static int R = 30;
    int framesTracked;

    bool pretraining;
    bool useFilter;
    bool useObjectness;
    bool scalePrior;

    bool updateTracker;
    int updateEveryNframes;
    const static int seed = 1;

    friend std::ostream& operator<<(std::ostream&, const Struck&);

    void turnOffPreTraining() {
        this->pretraining = false;
    }
    int display;

    std::vector<double> edge_params;
    std::vector<double> straddling_params;

    std::vector<cv::Rect> boundingBoxes;

    Struck() {}
    void setGroundTruthBox(cv::Rect box) {
        this->gtBox = box;
    }




    Struck(bool, bool, bool, bool, bool,
                             std::string,
                             std::string,
                             std::string);

    Struck(OLaRank_old* olarank_, Feature* feature_,
           LocationSampler* samplerSearch_, LocationSampler* samplerUpdate_,
           bool useObjectness_, bool scalePrior_, bool useFilter_,
           int usePretraining_, int display_) {
        olarank = olarank_;
        feature = feature_;
        samplerForSearch = samplerSearch_;
        samplerForUpdate = samplerUpdate_;
        useObjectness = useObjectness_;
        scalePrior = scalePrior_;
        useFilter = useFilter_;
        display = display_;
        pretraining = usePretraining_;
        objPlot = new Plot(500);
        framesTracked = 0;
        updateEveryNframes = 3;
        updateTracker = true;

    }


    void setParameters(OLaRank_old* olarank_, Feature* feature_,
                       LocationSampler* samplerSearch_,
                       LocationSampler* samplerUpdate_, bool useFilter_,
                       int display_){
        this->olarank = olarank_;
        this->feature = feature_;
        this->samplerForSearch = samplerSearch_;
        this->samplerForUpdate = samplerUpdate_;
        this->useFilter = useFilter_;
        this->display = display_;
    }


    cv::Mat getObjectnessCanvas() {
        return objectnessCanvas;
    }

    void setUpdateNFrames(int n){
        this->updateEveryNframes = n;
    }

    void setBudget(int B){
        this->olarank->B = B;
    }

    void setObjectnessCanvas(cv::Mat c) {
        this->objectnessCanvas = c;
    }


    static Struck getTracker();
    static Struck getTracker(bool, bool, bool, bool, bool);
    static Struck getTracker(bool, bool, bool, bool, bool,
                             std::string, std::string);

    static Struck getTracker(bool, bool, bool, bool, bool,
                             std::string,
                             std::string,
                             std::string);

    std::vector<cv::Rect> getBoundingBoxes() {
        return this->boundingBoxes;
    }

    void videoCapture();

    void initialize(std::string image_name, int x, int y, int width,
                    int height) {
        cv::Rect b(x, y, width, height);
        cv::Mat image = cv::imread(image_name, 0);
        this->initialize(image, b);
    }

    void initialize(cv::Mat& image, cv::Rect& location);

    void initialize(cv::Mat& image, cv::Rect& location,
                    int updateEveryNFrames,double b, int P, int R, int Q);

    void allocateCanvas(cv::Mat&);

    virtual cv::Rect track(cv::Mat& image);

    virtual cv::Rect track(std::string image_name) {
        cv::Mat image = cv::imread(image_name);
        return this->track(image);
    }

    virtual void setParams(const std::unordered_map<std::string, double>& map) {
    }

   virtual void setFeatureParams(const std::unordered_map<std::string, std::string> & map) {
        this->feature->setParams(map);
    }

    void updateDebugImage(cv::Mat* canvas,cv::Mat& img,
                          cv::Rect &bestLocation,cv::Scalar colorOfBox);

    void applyTrackerOnDataset(Dataset* dataset,std::string rootFolder,
                               std::string saveFolder, bool saveResults);

    void applyTrackerOnVideo(Dataset* dataset, std::string rootFolder,
                             int videoNumber);


    void saveResults(string fileName);

    void copyFromRectangleToImage(cv::Mat& canvas,
                                  cv::Mat& image,cv::Rect rect,
                                  int step,cv::Vec3b color);

    std::vector<cv::Rect> initializeBeforeFeatureExtraction(const cv::Mat &image,
                                                            cv::Rect &location,
                                                            int updateEveryNFrames,
                                                            double b, int P,
                                                            int R, int );

    void initializeAfterFeatureExtraction(const cv::Mat& features,
                                           std::vector<cv::Rect>& locations,
                                           cv::Mat& image){
        cv::Mat y = this->feature->reshapeYs(locations);
        this->olarank->initialize(features, y, 0, framesTracked);
        if (display == 1) {
            cv::Scalar color(0, 255, 0);
            cv::Mat plotImg = image.clone();

            cv::rectangle(plotImg, lastLocation, color, 2);
            cv::imshow("Tracking window", plotImg);
            this->objectnessCanvas = plotImg;
            cv::waitKey(1);

        } else if (display == 2) {

            this->allocateCanvas(image);
            this->frames.insert({framesTracked, image});
            this->updateDebugImage(&this->canvas, image, this->lastLocation,
                                   cv::Scalar(250, 0, 0));
        } else if (display == 3) {

            // initalize here...
            this->objPlot->initialize();
        }

        framesTracked++;
    }

    void preTraining(cv::Mat&,const cv::Rect& location);


    void reset(){

        this->olarank->clear();
        this->boundingBoxes.clear();
        this->frames.clear();
        this->framesTracked=0;
    };

    ~Struck(){
        this->reset();
        delete olarank;
        delete feature;
        delete samplerForSearch;
        delete samplerForUpdate;
        delete objPlot;
    };
};

#endif /* defined(__Robust_Struck__Struck__) */
