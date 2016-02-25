#ifndef SRC_TESTS_TEST_OLARANK_CPP_
#define SRC_TESTS_TEST_OLARANK_CPP_

#include <cmath>
#include <vector>
#include <glog/logging.h>
#include "TestOLaRank.h"

#include "../arma/arma_supportData.h"

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

bool TestOLaRank::isSupportVectorEqual(supportData *s,
                                       armadillo::supportData *arma_s) {
    bool equal = true;

    if (!(isEqual(*arma_s->x, s->x)))
        equal = false;

    if (!(isEqual(*arma_s->y, s->y)))
        equal = false;

    if (!(isEqual(*arma_s->beta, s->beta)))
        equal = false;

    if (!(isEqual(*arma_s->grad, s->grad)))
        equal = false;

    if (arma_s->label != s->label)
        equal = false;

    if (arma_s->frameNumber != s->frameNumber)
        equal = false;

    return equal;
}

bool TestOLaRank::isSetOLaRankSetS_equal() {
    int num = olarank->S.size();
    int arma_num = arma_olarank->S.size();

    if (num != arma_num)
        return false;

    for (int i = 0; i < num; i++) {
        if (!isSupportVectorEqual(olarank->S[i], arma_olarank->S[i]))
            return false;
    }

    return true;
}

std::vector<cv::Rect> TestOLaRank::getLocations(const cv::Mat &image,
                                                int boxes) {
    int r = image.rows;
    int c = image.cols;
    CHECK_GT(boxes, 0);
    int count = 0;

    std::vector<cv::Rect> locations;
    while (count < boxes) {
        cv::Mat cvbox = draw->getRandomBoundingBox(r, c);
        cv::Rect box(cvbox.at<double>(0, 1), cvbox.at<double>(0, 2),
                     cvbox.at<double>(0, 3), cvbox.at<double>(0, 4));

        locations.push_back(box);
        count++;
    }

    return locations;
}

void TestOLaRank::performProcessNewAndSMOAndbudget() {
    boost::tuple<cv::Mat, cv::Mat, int, int> processNewData =
        this->prepareForProcessNew();

    cv::Mat newX = boost::get<0>(processNewData);
    cv::Mat y_hat = boost::get<1>(processNewData);
    int label = boost::get<2>(processNewData);
    int frameNumber = boost::get<3>(processNewData);

    arma::mat arma_newX = convertCVToMat(newX);
    arma::mat arma_y_hat = convertCVToMat(y_hat);

    boost::tuple<cv::Mat, cv::Mat, cv::Mat> p_new =
        this->olarank->processNew(newX, y_hat, label, frameNumber);

    boost::tuple<arma::mat, arma::mat, arma::mat> arma_p_new =
        this->arma_olarank->processNew(arma_newX, arma_y_hat, label,
                                       frameNumber);

    cv::Mat y_plus, y_neg;
    cv::Mat grad = cv::Mat::zeros(1, newX.rows, CV_64F);
    boost::tie(y_plus, y_neg, grad) = p_new;

    // add new element into set S
    supportData *support =
        new supportData(newX, y_hat, label, 0, newX.rows, frameNumber);
    (support->grad) = grad;

    // delete support;

    double i = this->olarank->S.size();
    this->olarank->S.push_back(support);

    arma::mat mat_y_plus, mat_y_neg;
    arma::mat mat_grad(1, arma_newX.n_rows, arma::fill::zeros);
    boost::tie(mat_y_plus, mat_y_neg, mat_grad) = arma_p_new;

    // add new element into set S
    armadillo::supportData *arma_support = new armadillo::supportData(
        arma_newX, arma_y_hat, label, arma_newX.size(), newX.rows, frameNumber);
    (*arma_support->grad) = mat_grad;
    this->arma_olarank->S.push_back(arma_support);

    assert(isSetOLaRankSetS_equal());

    this->olarank->smoStep(i, y_plus, y_neg);
    this->arma_olarank->smoStep(i, mat_y_plus, mat_y_neg);

    assert(isSetOLaRankSetS_equal());

    this->olarank->budgetMaintance();
    this->arma_olarank->budgetMaintance();

    assert(isSetOLaRankSetS_equal());
}

TEST_F(TestOLaRank, testProcessOld) {
    this->performProcessNewAndSMOAndbudget();

    double i;
    cv::Mat y_plus, y_neg;
    boost::tuple<double, cv::Mat, cv::Mat> p_old = this->olarank->processOld();

    boost::tie(i, y_plus, y_neg) = p_old;

    arma::mat arma_y_plus, arma_y_neg;
    boost::tuple<double, arma::mat, arma::mat> arma_p_old =
        this->arma_olarank->processOld();

    boost::tie(i, arma_y_plus, arma_y_neg) = arma_p_old;

    assert(isSetOLaRankSetS_equal());

    this->olarank->smoStep(i, y_plus, y_neg);
    this->arma_olarank->smoStep(i, arma_y_plus, arma_y_neg);
    assert(isSetOLaRankSetS_equal());

    this->olarank->budgetMaintance();
    this->arma_olarank->budgetMaintance();
    assert(isSetOLaRankSetS_equal());
}

TEST_F(TestOLaRank, testOptimize) {
    this->performProcessNewAndSMOAndbudget();

    double i;
    cv::Mat y_plus, y_neg;
    boost::tuple<double, cv::Mat, cv::Mat> p_old = this->olarank->processOld();

    boost::tie(i, y_plus, y_neg) = p_old;

    arma::mat arma_y_plus, arma_y_neg;
    boost::tuple<double, arma::mat, arma::mat> arma_p_old =
        this->arma_olarank->processOld();

    double arma_i;
    boost::tie(arma_i, arma_y_plus, arma_y_neg) = arma_p_old;

    assert(isSetOLaRankSetS_equal());

    this->olarank->smoStep(i, y_plus, y_neg);
    this->arma_olarank->smoStep(i, arma_y_plus, arma_y_neg);
    assert(isSetOLaRankSetS_equal());

    this->olarank->budgetMaintance();
    this->arma_olarank->budgetMaintance();
    assert(isSetOLaRankSetS_equal());


    boost::tuple<double, cv::Mat, cv::Mat> optimize = this->olarank->optimize();
    boost::tie(i, y_plus, y_neg) = optimize;

    boost::tuple<double, arma::mat, arma::mat> arma_optimize =
        this->arma_olarank->optimize();

    boost::tie(arma_i, arma_y_plus, arma_y_neg) = arma_optimize;

    assert(i == arma_i);
    assert(isEqual(arma_y_plus, y_plus));
    assert(isEqual(arma_y_neg, y_neg));

    assert(isSetOLaRankSetS_equal());
    this->olarank->smoStep(i, y_plus, y_neg);
    this->arma_olarank->smoStep(arma_i, arma_y_plus, arma_y_neg);

    assert(isSetOLaRankSetS_equal());
}

boost::tuple<cv::Mat, cv::Mat, int, int>
TestOLaRank::prepareForProcessNew(int boxes) {
    cv::Mat image = draw->getRandomImage();

    std::vector<cv::Rect> locations = this->getLocations(image, boxes);

    cv::Mat processedImage = this->feature->prepareImage(&image);
    cv::Mat x = this->feature->calculateFeature(processedImage, locations);
    cv::Mat y = this->feature->reshapeYs(locations);

    arma::mat arma_x = convertCVToMat(x);
    arma::mat arma_y = convertCVToMat(y);

    int m = this->feature->calculateFeatureDimension();

    int label = 0;
    int frameNumber = 0;
    supportData *s1 = new supportData(x, y, label, m, x.rows, frameNumber);

    armadillo::supportData *arma_s1 = new armadillo::supportData(
        arma_x, arma_y, label, m, x.rows, frameNumber);

    this->olarank->S.push_back(s1);
    this->arma_olarank->S.push_back(arma_s1);

    boost::tuple<cv::Mat, cv::Mat, int, int> result =
        boost::make_tuple(x, y, label, frameNumber);

    return result;
}

TEST_F(TestOLaRank, smoStep) {
    boost::tuple<cv::Mat, cv::Mat, int, int> processNewData =
        this->prepareForProcessNew();

    cv::Mat newX = boost::get<0>(processNewData);
    cv::Mat y_hat = boost::get<1>(processNewData);
    int label = boost::get<2>(processNewData);
    int frameNumber = boost::get<3>(processNewData);

    arma::mat arma_newX = convertCVToMat(newX);
    arma::mat arma_y_hat = convertCVToMat(y_hat);

    boost::tuple<cv::Mat, cv::Mat, cv::Mat> p_new =
        this->olarank->processNew(newX, y_hat, label, frameNumber);

    boost::tuple<arma::mat, arma::mat, arma::mat> arma_p_new =
        this->arma_olarank->processNew(arma_newX, arma_y_hat, label,
                                       frameNumber);

    cv::Mat y_plus, y_neg;
    cv::Mat grad = cv::Mat::zeros(1, newX.rows, CV_64F);
    boost::tie(y_plus, y_neg, grad) = p_new;

    // add new element into set S
    supportData *support =
        new supportData(newX, y_hat, label, 0, newX.rows, frameNumber);
    (support->grad) = grad;

    // delete support;

    double i = this->olarank->S.size();
    this->olarank->S.push_back(support);

    arma::mat mat_y_plus, mat_y_neg;
    arma::mat mat_grad(1, arma_newX.n_rows, arma::fill::zeros);
    boost::tie(mat_y_plus, mat_y_neg, mat_grad) = arma_p_new;

    // add new element into set S
    armadillo::supportData *arma_support = new armadillo::supportData(
        arma_newX, arma_y_hat, label, arma_newX.size(), newX.rows, frameNumber);
    (*arma_support->grad) = mat_grad;
    this->arma_olarank->S.push_back(arma_support);

    assert(isSetOLaRankSetS_equal());

    this->olarank->smoStep(i, y_plus, y_neg);
    this->arma_olarank->smoStep(i, mat_y_plus, mat_y_neg);

    assert(isSetOLaRankSetS_equal());

    this->olarank->budgetMaintance();
    this->arma_olarank->budgetMaintance();

    assert(isSetOLaRankSetS_equal());
}

TEST_F(TestOLaRank, processNew) {
    boost::tuple<cv::Mat, cv::Mat, int, int> processNewData =
        this->prepareForProcessNew();

    cv::Mat x = boost::get<0>(processNewData);
    cv::Mat y = boost::get<1>(processNewData);
    int label = boost::get<2>(processNewData);
    int frameNumber = boost::get<3>(processNewData);

    arma::mat arma_x = convertCVToMat(x);
    arma::mat arma_y = convertCVToMat(y);

    boost::tuple<cv::Mat, cv::Mat, cv::Mat> processNewOutput =
        this->olarank->processNew(x, y, label, frameNumber);

    boost::tuple<arma::mat, arma::mat, arma::mat> arma_processNewOutput =
        this->arma_olarank->processNew(arma_x, arma_y, label, frameNumber);

    cv::Mat r_1 = boost::get<0>(processNewOutput);
    cv::Mat r_2 = boost::get<1>(processNewOutput);
    cv::Mat r_3 = boost::get<2>(processNewOutput);

    arma::mat arma_r_1 = boost::get<0>(arma_processNewOutput);
    arma::mat arma_r_2 = boost::get<1>(arma_processNewOutput);
    arma::mat arma_r_3 = boost::get<2>(arma_processNewOutput);

    assert(isEqual(arma_r_1, r_1));
    assert(isEqual(arma_r_2, r_2));
    assert(isEqual(arma_r_3, r_3));
}

TEST_F(TestOLaRank, calculate_kernel) {
    cv::Mat cvf1 = draw->getRandomFloatMatrix();
    cv::Mat cvf2 = draw->getRandomFloatMatrix();

    arma::mat armaf1 = convertCVToMat(cvf1);
    arma::mat armaf2 = convertCVToMat(cvf2);

    int r = draw->window_height;
    int c = draw->window_width;

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            ASSERT_NEAR(olarank->calculate_kernel(cvf1, i, cvf2, j),
                        arma_olarank->calculate_kernel(armaf1, i, armaf2, j),
                        this->PRECISION);
        }
    }
}

TEST_F(TestOLaRank, kernel_fast) {
    cv::Mat cvf1 = draw->getRandomFloatMatrix();
    cv::Mat cvf2 = draw->getRandomFloatMatrix();

    arma::mat armaf1 = convertCVToMat(cvf1);
    arma::mat armaf2 = convertCVToMat(cvf2);

    int r = draw->window_height;
    int c = draw->window_width;

    cv::Mat cvBox1 = draw->getRandomBoundingBox(r, c);
    cv::Mat cvBox2 = draw->getRandomBoundingBox(r, c);

    CHECK(cvBox1.at<double>(0, 1) + cvBox1.at<double>(0, 3) < r);
    CHECK(cvBox1.at<double>(0, 2) + cvBox1.at<double>(0, 4) < c);

    CHECK(cvBox2.at<double>(0, 1) + cvBox2.at<double>(0, 3) < r);
    CHECK(cvBox2.at<double>(0, 2) + cvBox2.at<double>(0, 4) < c);

    arma::mat armaBox1 = convertCVToMat(cvBox1);
    arma::mat armaBox2 = convertCVToMat(cvBox2);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++) {
            // The implementation is not using cvBox1, cvBox2, armaBox1,
            // armaBox2 arugments
            ASSERT_NEAR(
                olarank->kernel_fast(cvf1, cvBox1, i, -1, cvf2, cvBox2, j, -2),
                arma_olarank->kernel_fast(armaf1, armaBox1, i, -1, armaf2,
                                          armaBox2, j, -2),
                this->PRECISION);
        }
    }
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
