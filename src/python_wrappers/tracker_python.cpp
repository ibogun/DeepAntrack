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

#include <boost/python.hpp>

class DeepStruck {
 public:
  Struck* tracker;


  boost::python::list track(std::string filename) {
    boost::python::list output;
    cv::Rect r = this->tracker->track(filename);

    output.append(r.x);
    output.append(r.y);
    output.append(r.width);
    output.append(r.height);
    return output;
  }

  void initialize(std::string filename, int x, int y,
                  int width, int height) {
    this->tracker->initialize(filename, x, y, width, height);
  }


  ~DeepStruck() {
    delete tracker;
  }

  void createTracker(std::string kernel, std::string feature, int filter) {

    bool pretraining = false;
    bool useEdgeDensity = false;
    bool useStraddling = false;
    bool scalePrior = false;

    std::string note = "Struck tracker";

    bool useFilter = false;

    this->tracker = new Struck(pretraining, useFilter, useEdgeDensity,
                               useStraddling, scalePrior, kernel, feature,
                               note);
  }

  void setDisplay(int display) {
    CHECK_NOTNULL(tracker);
    this->tracker->display = display;
  }

  void killDisplay() {
    cv::destroyAllWindows();
  }
};

using namespace boost::python;

BOOST_PYTHON_MODULE(DeepAntrack)

{

  class_<DeepStruck>("DeepStruck")
    .def("initialize", &DeepStruck::initialize)
    .def("createTracker", &DeepStruck::createTracker)
    .def("track", &DeepStruck::track)
    .def("setDisplay", &DeepStruck::setDisplay)
    .def("killDisplay", &DeepStruck::killDisplay)
    ;

}
// find how to write functions which return some values in c++/python boost
// framework
