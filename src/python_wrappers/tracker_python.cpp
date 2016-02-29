#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>
#include "../../src/Tracker/OLaRank_old.h"
#include "../../src/Tracker/LocationSampler.h"

#include "../../src/Tracker/AllTrackers.h"
#include "../../src/Kernels/AllKernels.h"
#include "../../src/Features/AllFeatures.h"
#include "numpy-opencv-converter/np_opencv_converter.hpp"
#include <boost/python.hpp>

class DeepStruck {
  public:
    Struck *tracker;

    std::vector<cv::Rect> rects;
    cv::Mat currentImage;

    const int maxFeatures = 100;

    // INITIALIZE #1
    void initializeBefore(std::string filename, int x, int y, int width,
                          int height) {

        cv::Mat image = cv::imread(filename);
        cv::Rect location(x, y, width, height);

        this->currentImage = image.clone();
        srand(tracker->seed);
        // set dimensions of the sampler

        tracker->updateEveryNframes = 3;
        // NOW
        int m = image.rows;
        int n = image.cols;

        tracker->samplerForSearch->setDimensions(n, m, location.height,
                                                 location.width);
        tracker->samplerForUpdate->setDimensions(n, m, location.height,
                                                 location.width);

        tracker->boundingBoxes.push_back(location);
        tracker->lastLocation = location;
        // sample in polar coordinates first

        std::vector<cv::Rect> locations;

        // add ground truth
        locations.push_back(location);
        tracker->samplerForUpdate->sampleEquiDistant(location, locations);

        this->rects = locations;
    }

    boost::python::list getImages() {
        boost::python::list out;
        for (int i = 0; i < this->rects.size(); i++) {
            cv::Mat im(this->currentImage, this->rects[i]);
            out.append(im);
        }
        return out;
    }

    // INITIALIZE #3
    void initializeAfter(const cv::Mat &x) {
        cv::Mat y = tracker->feature->reshapeYs(this->rects);
        LOG(INFO) << "Prior to olarank->initialize";
        LOG(INFO) << "Rows: " << x.rows;
        LOG(INFO) << "Cols: " << x.cols;
        LOG(INFO) << "Type: " << x.type();
        LOG(INFO) << "Rows: " << y.rows;
        LOG(INFO) << "Cols: " << y.cols;

        tracker->olarank->initialize(x, y, 0, tracker->framesTracked);
        LOG(INFO) << "AFTER olarank->initialize";
        // add images, in case we want to show support vectors

        if (tracker->display == 1) {
            cv::Scalar color(0, 255, 0);
            cv::Mat plotImg = this->currentImage.clone();

            cv::rectangle(plotImg, tracker->lastLocation, color, 2);
            cv::imshow("Tracking window", plotImg);
            tracker->objectnessCanvas = plotImg;
            cv::waitKey(1);
        }

        tracker->framesTracked++;
    }

    void trackDetectBefore(std::string filename) {
        cv::Mat image = cv::imread(filename);

        this->currentImage = image.clone();
        std::vector<cv::Rect> locationsOnaGrid;

        locationsOnaGrid.push_back(tracker->lastLocation);

        tracker->samplerForSearch->sampleEquiDistantMultiScale(
            tracker->lastLocation, locationsOnaGrid);

        this->rects = locationsOnaGrid;
    }

    boost::python::list trackDetectAfter(const cv::Mat &x) {
        std::vector<double> predictions = tracker->olarank->predictAll(x);
        int groundTruth = 0;
        double maxElement = predictions[0];
        for (int i = 0; i < predictions.size(); i++) {
            if (maxElement < predictions[i]) {
                maxElement = predictions[i];
                groundTruth = i;
            }
        }
        cv::Rect bestLocationDetector = this->rects[groundTruth];
        tracker->boundingBoxes.push_back(bestLocationDetector);
        tracker->lastLocation = bestLocationDetector;

        boost::python::list output;
        cv::Rect r = bestLocationDetector;

        output.append(r.x);
        output.append(r.y);
        output.append(r.width);
        output.append(r.height);
        return output;
    }

    void trackUpdateBefore() {
        if (tracker->updateTracker &&
            tracker->boundingBoxes.size() % tracker->updateEveryNframes == 0) {

            // sample for updating the tracker
            std::vector<cv::Rect> locationsOnPolarPlane;
            locationsOnPolarPlane.push_back(tracker->lastLocation);

            tracker->samplerForUpdate->sampleEquiDistant(tracker->lastLocation,
                                                         locationsOnPolarPlane);

            this->rects = locationsOnPolarPlane;
        }
    }

    void trackUpdateAfter(const cv::Mat &x_update) {
        cv::Mat y_update = tracker->feature->reshapeYs(this->rects);
        tracker->olarank->process(x_update, y_update, 0,
                                  tracker->framesTracked);

        if (tracker->display == 1) {
            cv::Scalar color(255, 0, 0);
            cv::Mat plotImg = currentImage.clone();
            cv::rectangle(plotImg, tracker->lastLocation, color, 2);

            cv::imshow("Tracking window", plotImg);
            cv::waitKey(1);
            tracker->objectnessCanvas = plotImg;
        }

        tracker->framesTracked++;
    }

    ~DeepStruck() { delete tracker; }

    void createTracker(std::string kernel, std::string feature, int filter) {

        bool pretraining = false;
        bool useEdgeDensity = false;
        bool useStraddling = false;
        bool scalePrior = false;

        std::string note = "Struck tracker";

        bool useFilter = false;

        this->tracker =
            new Struck(pretraining, useFilter, useEdgeDensity, useStraddling,
                       scalePrior, kernel, feature, note);
    }

    void setDisplay(int display) {
        CHECK_NOTNULL(tracker);
        this->tracker->display = display;
    }

    void killDisplay() { cv::destroyAllWindows(); }
};

cv::Mat cloneImage(const cv::Mat &image) {
    cv::Mat out;
    image.convertTo(out, CV_64F);
    return out;
}

using namespace boost::python;

BOOST_PYTHON_MODULE(DeepAntrack)

{
    boost::python::def("cloneImage", &cloneImage);

    class_<DeepStruck>("DeepStruck")
        .def("initializeBefore", &DeepStruck::initializeBefore)
        .def("getImages", &DeepStruck::getImages)
        .def("initializeAfter", &DeepStruck::initializeAfter)
        .def("trackDetectBefore", &DeepStruck::trackDetectBefore)
        .def("trackDetectAfter", &DeepStruck::trackDetectAfter)
        .def("trackUpdateBefore", &DeepStruck::trackUpdateBefore)
        .def("trackUpdateAfter", &DeepStruck::trackUpdateAfter)
        .def("createTracker", &DeepStruck::createTracker)
        .def("setDisplay", &DeepStruck::setDisplay)
        .def("killDisplay", &DeepStruck::killDisplay);
}
// find how to write functions which return some values in c++/python boost
// framework
