#ifndef SRC_TESTS_TEST_OLARANK_CPP_
#define SRC_TESTS_TEST_OLARANK_CPP_

#include "TestOLaRank.h"
#include <glog/logging.h>
#include <cmath>

bool isEqual(const arma::mat &m1, const cv::Mat &m2) {
    int rows = m1.n_rows;
    int cols = m1.n_cols;

    int cv_rows = m2.rows;
    int cv_cols = m2.cols;

    if (rows != cv_rows)
        return false;
    if (cols != cv_cols)
        return false;

    const double precision = 0.0001;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (abs(m1(i, j) - m2.at<double>(i, j)) > precision) {
                return false;
            }
        }
    }
    return true;
}

cv::Mat convertMatToCV(const arma::mat &m1) {
    cv::Mat c = cv::Mat::zeros(m1.n_rows, m1.n_cols, CV_64F);
    int count = 0;
    for (int i = 0; i < m1.n_rows; i++) {
        for (int j = 0; j < m1.n_cols; j++) {
            c.at<double>(i, j) = m1(i, j);
        }
    }
    return c;
}

arma::mat convertCVToMat(const cv::Mat &m1) {
    arma::mat c = arma::zeros(m1.rows, m1.cols);

    uchar depth = m1.type() & CV_MAT_DEPTH_MASK;
    CHECK(depth == CV_64F);
    for (int i = 0; i < m1.rows; i++) {
        for (int j = 0; j < m1.cols; j++) {
            c(i, j) = m1.at<double>(i, j);
        }
    }
    return c;
}

TEST_F(TestOLaRank, matToCVConversion) {
    cv::Mat image = draw->getRandomFloatMatrix();

    arma::mat m = convertCVToMat(image);
    cv::Mat image_back = convertMatToCV(m);
    assert(true == isEqual(m, image_back));
}

TEST_F(TestOLaRank, loss) {
    cv::Mat cv_y = cv::Mat::zeros(1, 5, CV_64F);
    cv::Mat cv_yhat = cv::Mat::zeros(1, 5, CV_64F);

    arma::mat arma_y = arma::zeros(1, 5);
    arma::mat arma_yhat = arma::zeros(1, 5);

    cv_y.at<double>(0, 1) = 5;
    cv_y.at<double>(0, 2) = 10;
    cv_y.at<double>(0, 3) = 3;
    cv_y.at<double>(0, 4) = 7;

    arma_y(0, 1) = 5;
    arma_y(0, 2) = 10;
    arma_y(0, 3) = 3;
    arma_y(0, 4) = 7;

    cv_yhat.at<double>(0, 1) = 7;
    cv_yhat.at<double>(0, 2) = 9;
    cv_yhat.at<double>(0, 3) = 6;
    cv_yhat.at<double>(0, 4) = 4;

    arma_yhat(0, 1) = 7;
    arma_yhat(0, 2) = 9;
    arma_yhat(0, 3) = 6;
    arma_yhat(0, 4) = 4;

    ASSERT_NE(0, olarank->loss(cv_y, cv_yhat));
    ASSERT_NEAR(olarank->loss(cv_y, cv_yhat),
                arma_olarank->loss(arma_y, arma_yhat), PRECISION);
    ASSERT_NEAR(olarank->loss(cv_yhat, cv_y),
                arma_olarank->loss(arma_yhat, arma_y), PRECISION);
}

#endif
