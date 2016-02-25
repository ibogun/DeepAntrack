

#ifndef SRC_TESTS_TEST_OLARANK_H_
#define SRC_TESTS_TEST_OLARANK_H_

#include "gtest/gtest.h"
#include "../arma/arma_Kernel.h"
#include "../arma/arma_LinearKernel.h"
#include "../arma/arma_OLaRank_old.h"
#include "DrawRandomImage.h"


#include "boost/tuple/tuple.hpp"
#include "../Kernels/Kernel.h"
#include "../Kernels/LinearKernel.h"
#include "../Tracker/OLaRank_old.h"

#include "../Features/Feature.h"
#include "../Features/RawFeatures.h"


class TestOLaRank : public ::testing::Test {



protected:
  OLaRank_old* olarank;
  armadillo::OLaRank_old* arma_olarank;
   const double PRECISION = 0.00001;
  DrawRandomImage* draw;
  Feature* feature;

  TestOLaRank() {
          Kernel* kernel = new LinearKernel;
          armadillo::Kernel* arma_kernel = new armadillo::LinearKernel;

          int dims = 4;

          feature = new RawFeatures(4);
          arma_olarank = new armadillo::OLaRank_old(arma_kernel);
          olarank = new OLaRank_old(kernel);
          // params
          int verbose = 0;
          int B = 10;
          int m = feature->calculateFeatureDimension();

          armadillo::params arma_p;
          arma_p.C = 100;
          arma_p.n_O = 10;
          arma_p.n_R = 10;

          params p;
          p.C = 100;
          p.n_O = 10;
          p.n_R = 10;

          olarank->setParameters(p, B, m , verbose);
          arma_olarank->setParameters(arma_p, B, m, verbose);
          int seed = 100;
          draw = new DrawRandomImage(seed);


  }
  virtual ~TestOLaRank(){
          delete arma_olarank;
          delete olarank;
          delete draw;
          delete feature;
          };
  virtual void SetUp(){};
  virtual void TearDown(){};
  std::vector<cv::Rect> getLocations(const cv::Mat& image, int n);

  boost::tuple<cv::Mat, cv::Mat, int, int> prepareForProcessNew(int boxes = 100);
  bool isSetOLaRankSetS_equal();
  bool isSupportVectorEqual(supportData* s, armadillo::supportData* arma_s);

  void performProcessNewAndSMOAndbudget();

};

#endif
