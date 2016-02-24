//
//  OLaRank_old.cpp
//  Structured_BING
//
//  Created by Ivan Bogun on 7/17/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include <vector>
#include "OLaRank_old.h"
#include <tuple>
#include "armadillo"
#include <math.h>
#include <assert.h>
#include <iostream>
#include <algorithm> // std::random_shuffle      // std::vector
#include <ctime>     // std::time
#include <cstdlib>
#include <cfloat>

using namespace std;

OLaRank_old::OLaRank_old(Kernel *svm_kernel_, int seed) {
  srand(seed);
  svm_kernel = svm_kernel_;
  // this->kern=new unordered_map<Key,arma::mat,KeyHash, KeyEqual>;
}

/***
 *  Set parameters. Used to replace constructor with parameters.
 *
 * @param learningParams
 * @param balance
 * @param m_
 * @param K_
 * @param verbose_
 */
void OLaRank_old::setParameters(params &learningParams, int &balance, int &m_,
                                int &verbose_) {
  parameters = learningParams;
  m = m_;
  B = balance;
  verbose = verbose_;
}

std::vector<double> OLaRank_old::predictAll(const cv::Mat &newX) {
  return this->svm_kernel->predictAll(newX, this->S, this->B);
}

/**
 * The function checks if constraints in multi-class SVM are satisfied
 */
void OLaRank_old::checkIfConstraintsSatisfied() {
  double sum = 0;
  for (int i = 0; i < this->S.size(); ++i) {
    sum = 0;
    for (int y = 0; y < this->S[i]->x.rows; ++y) {
      double beta_y = this->S[i]->beta.at<double>(0, y);
      assert(beta_y <= this->parameters.C * (this->S[i]->label == y));
      sum += beta_y;
    }

    if (abs(sum) > 0.00001) {
      cout << "Pattern: " << i << " out of " << this->S.size() << "\n";
      cout << this->S[i]->beta;
      cout << abs(sum) << " \n";
    }

    assert(abs(sum) < 0.00001);
  }
}

/**
 * The function calculates gradient for newly added pattern and a label; it also
 * produces (y_plus,y_neg, gradient)
 *
 * @param newX			- 		pattern
 * @param y_hat 		- 		label ( set of possible
 * positions)
 * @param label 		- 		predicted position
 * @return 				- 		triplet
 * (y_plus,y_neg,gradient)
 */
tuple<cv::Mat, cv::Mat, cv::Mat> OLaRank_old::processNew(const cv::Mat &newX,
                                                         const cv::Mat &y_hat,
                                                         int label,
                                                         int frameNumber) {
  // calculate gradient for the new example

  cv::Mat y_bar_mat;

  int K = newX.rows;

  cv::Mat grad = cv::Mat::zeros(1, K, CV_64F);

  for (int y = 0; y < K; ++y) {
    grad.at<double>(y) -= this->loss(y_hat.row(label), y_hat.row(y));
    for (int i = 0; i < this->S.size(); ++i) {
      for (int y_bar = 0; y_bar < this->S[i]->x.rows; ++y_bar) {
        supportData *S_i = this->S[i];

        double beta_y_bar = S_i->beta.at<double>(y_bar);
        if (beta_y_bar != 0) {
          grad.at<double>(y) -=
              beta_y_bar *
              this->kernel_fast(this->S[i]->x, this->S[i]->y,
                                (this->S[i]->y).row(y_bar).at<double>(0),
                                (this->S[i])->frameNumber, newX, y_hat,
                                y_hat.row(y).at<double>(0), frameNumber);
        }
      }
    }
  }

  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;

  minMaxLoc(grad, &minVal, &maxVal, &minLoc, &maxLoc);
  // grad.min(y_neg_idx);
  int y_neg_idx = minLoc.y;

  if (this->verbose > 1) {
    cout << "ProcessNew step: \n";
    cout << "(y_plus,y_neg) " << y_hat.row(label)(0) << " "
         << y_hat.row(y_neg_idx)(0) << endl;
    cout << "--------------------" << endl;
  }

  // tuple<mat, mat, mat> result = make_tuple(y_plus, y_neg, grad);
  tuple<cv::Mat, cv::Mat, cv::Mat> result =
      make_tuple(y_hat.row(label), y_hat.row(y_neg_idx), grad);
  return result;
}

/**
 * Function which maintains that the number of support patterns is less or equal
 * to the balance (variable B).
 */
int OLaRank_old::budgetMaintance() {

    if( this->S.size() < this->B) return -1;
  int test = 0;

  vector<int> idxToBeDeleted;

  for (int i = 0; i < this->S.size(); ++i) {

    int sum = 0;

    for (int j = 0; j < this->S[i]->x.rows; ++j) {
      // if (abs((*this->S[i]->beta)(j))>=1.0e-7) {

      if (abs((this->S[i]->beta).at<double>(j)) != 0) {
        test += 1;
      } else {
        sum++;
      }
    }
    // cout<<"Idx: "<<i<<" zeros: "<<sum<<endl;
    if (sum == this->S[i]->x.rows) {
      idxToBeDeleted.push_back(i);
    }
  }
  if (idxToBeDeleted.size() > 1) {
    std::sort(idxToBeDeleted.begin(), idxToBeDeleted.end());
  }

  // be
  auto beginningIterator = this->S.begin();
  for (int i = (int)idxToBeDeleted.size() - 1; i >= 0; i--) {
    // cout<<"deleting idx: "<<idxToBeDeleted[i]<<endl;

    // delete kernel values
    deleteKernelValues(this->S[idxToBeDeleted[i]]->frameNumber);

    delete this->S[idxToBeDeleted[i]];
    this->S.erase(beginningIterator + idxToBeDeleted[i]);
  }
  int n = test;

  // If balance is not exceeded - return.
  if (n <= this->B) {
    return -1;
  }

  int maxNRows = 0;

  for (int i = 0; i < this->S.size(); i++) {
    if (this->S[i]->x.rows >= maxNRows) {
      maxNRows = this->S[i]->x.rows;
    }
  }

  cv::Mat scores =
      cv::Mat::zeros(static_cast<int>(this->S.size()), maxNRows, CV_64F);
  cv::Mat y_r, x_r;
  scores = scores * INFINITY;

  for (int i = 0; i < this->S.size(); ++i) {
    for (int k = 0; k < this->S[i]->x.rows; ++k) {
      if ((this->S[i]->beta).at<double>(k) < 0) {
        x_r = this->S[i]->x;
        y_r = this->S[i]->y;

        int S_i_framenumber = this->S[i]->frameNumber;

        scores.at<double>(i, k) =
            (pow((this->S[i]->beta).at<double>(k), 2)) *
            (this->kernel_fast(this->S[i]->x, y_r, y_r.row(k).at<double>(0),
                               S_i_framenumber, this->S[i]->x, y_r,
                               y_r.row(k).at<double>(0),
                               S_i_framenumber) +
             (this->kernel_fast(
                 this->S[i]->x,
                 y_r,
                 y_r.row(this->S[i]->label).at<double>(0),
                 S_i_framenumber,
                 this->S[i]->x,
                 y_r,
                 y_r.row(this->S[i]->label).at<double>(0),
                 S_i_framenumber)) -
             2 * this->kernel_fast(this->S[i]->x,
                                   y_r,
                                   y_r.row(this->S[i]->label).at<double>(0),
                                  S_i_framenumber,
                                   this->S[i]->x, y_r,
                                   y_r.row(k).at<double>(0),
                                   S_i_framenumber));
      }
    }
  }


  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;

  minMaxLoc( scores, &minVal, &maxVal, &minLoc, &maxLoc );

  int I = minLoc.x;
  int J = minLoc.y;
  int numberOfNegativeSP = 0;

  vector<int> negativeSP;
  vector<int> positiveSP;

  for (int i = 0; i < this->S[I]->x.rows; ++i) {
    if ((this->S[I]->beta).at<double>(i) < 0) {
      numberOfNegativeSP++;
      negativeSP.push_back(i);
    }

    if ((this->S[I]->beta).at<double>(i) > 0) {
      positiveSP.push_back(i);
    }
  }

  if (this->verbose >= 1) {
    cout << "Balancing step. Indices: I=" << I << " J==" << J << "\n";
    cout << "Max value " << scores.at<double>(I, J) << "\n";
  }

  if (numberOfNegativeSP == 1) {
    // if there is only one negative support vector - delete the whole pattern
    // there might be a number of positive vectors. Adjust gradients before
    // deleting the row.

    std::vector<int> allSPtobeDeleted;

    allSPtobeDeleted.reserve(positiveSP.size() +
                             negativeSP.size()); // preallocate memory
    allSPtobeDeleted.insert(allSPtobeDeleted.end(), positiveSP.begin(),
                            positiveSP.end());
    allSPtobeDeleted.insert(allSPtobeDeleted.end(), negativeSP.begin(),
                            negativeSP.end());

    for (int z = 0; z < allSPtobeDeleted.size(); ++z) {
      // delete beta(I,z)
      int idx = allSPtobeDeleted[z];
      for (int ii = 0; ii < this->S.size(); ++ii) {
        for (int yy = 0; yy < this->S[ii]->x.rows; ++yy) {
            (this->S[ii]->grad).at<double>(yy) +=
                this->S[I]->beta.at<double>(idx) * kernel_fast(
              this->S[ii]->x, this->S[ii]->y,
              (this->S[ii]->y).row(yy).at<double>(0),
              (this->S[ii])->frameNumber, this->S[I]->x, this->S[I]->y,
              (this->S[I]->y).row(idx).at<double>(0), (this->S[I])->frameNumber);
        }
      }
    }

    deleteKernelValues(this->S[I]->frameNumber);
    // Problem here;
    delete this->S[I];
    this->S.erase(this->S.begin() + (I));

    return I;
  } else {
    // delete negative support vector; in order to compensate (satisfy
    // sum(beta(I,:))=0)
    // add it to the positive support vector
    if ((this->S[I]->beta).at<double>(this->S[I]->label) < 0) {
      std::cout << "Should be positive support pattern is not positive"
                << std::endl;
    }

    std::vector<int> allSPtobeDeleted;
    allSPtobeDeleted.push_back(J);
    // for (int z=0; z<allSPtobeDeleted.size(); ++z) {
    // delete beta(I,z)
    int idx = J;
    for (int ii = 0; ii < this->S.size(); ++ii) {
      for (int yy = 0; yy < this->S[ii]->x.rows; ++yy) {
        (this->S[ii]->grad).at<double>(yy) += (this->S[I]->beta).at<double>(idx)
            *kernel_fast(
            this->S[ii]->x, this->S[ii]->y,
            (this->S[ii]->y).row(yy).at<double>(0),
            (this->S[ii])->frameNumber, this->S[I]->x, this->S[I]->y,
            (this->S[I]->y).row(idx).at<double>(0), (this->S[I])->frameNumber);
      }
    }

    idx = this->S[I]->label;
    for (int ii = 0; ii < this->S.size(); ++ii) {
      for (int yy = 0; yy < this->S[ii]->x.rows; ++yy) {
        (this->S[ii]->grad).at<double>(yy) -= (this->S[I]->beta).at<double>(J)
            *kernel_fast(
            this->S[ii]->x, (this->S[ii]->y),
            (this->S[ii]->y).row(yy).at<double>(0),
            (this->S[ii])->frameNumber, this->S[I]->x, this->S[I]->y,
            (this->S[I]->y).row(idx).at<double>(0), (this->S[I])->frameNumber);
      }
    }

    // std::cout<<"this changes everything"<<std::endl;

    (this->S[I]->beta).at<double>(this->S[I]->label) +=
        (this->S[I]->beta).at<double>(J);

    (this->S[I]->beta).at<double>(J) = 0;
  }

  if (n - 1 > this->B) {
    budgetMaintance();
  }

  return -1;
}

/**
 * SMO type step for Struck algorithm. All gradients will be updated in this
 * function, as	 well as beta for the i pattern.
 *
 * @param i  		- index of the pattern whose beta will be updated
 * @param y_plus 	- y plus
 * @param y_neg 	- y minus
 */
void OLaRank_old::smoStep(int i, const cv::Mat &y_plus,
                          const  cv::Mat &y_neg) {

  double k_00 = this->kernel_fast(
      this->S[i]->x, this->S[i]->y, y_plus.at<double>(0),
      (this->S[i])->frameNumber,
      this->S[i]->x, this->S[i]->y, y_plus.at<double>(0),
      (this->S[i])->frameNumber);

  double k_11 = this->kernel_fast(
      this->S[i]->x, this->S[i]->y,
      y_neg.at<double>(0), (this->S[i])->frameNumber,
      this->S[i]->x, this->S[i]->y,
      y_neg.at<double>(0), (this->S[i])->frameNumber);

  double k_01 = this->kernel_fast(
      this->S[i]->x, this->S[i]->y, y_plus.at<double>(0),
      (this->S[i])->frameNumber,
      this->S[i]->x, this->S[i]->y, y_neg.at<double>(0),
      (this->S[i])->frameNumber);

  double lambda_u =
      ((this->S[i]->grad).at<double>(y_plus.at<double>(0)) -
       (this->S[i]->grad).at<double>(y_neg.at<double>(0))) /
      (k_00 + k_11 - 2 * k_01);

  int y_plus_0 = y_plus.at<double>(0);
  int y_neg_0 = y_neg.at<double>(0);

  double tmp2 = this->parameters.C *
      static_cast<int>(y_plus_0 == this->S[i]->label) -
      (this->S[i]->beta).at<double>(y_plus_0);

  double const tmp = min(lambda_u, tmp2);

  double lambda = max(static_cast<double>(0), tmp);

  (this->S[i]->beta).at<double>(y_plus_0) += lambda;
  (this->S[i]->beta).at<double>(y_neg_0) -= lambda;

  double k_0, k_1;

  if (lambda != 0) {
    for (int j = 0; j < this->S.size(); ++j) {
      // for (int y = 0; y < this->K; ++y) {
      for (int y = 0; y < this->S[j]->x.rows; ++y) {

        k_0 = this->kernel_fast(
            this->S[j]->x, this->S[j]->y, (this->S[j]->y).row(y).at<double>(0),
            (this->S[j])->frameNumber, this->S[i]->x, this->S[i]->x,
            y_plus_0, (this->S[i])->frameNumber);
        k_1 = this->kernel_fast(
            this->S[j]->x, this->S[j]->y,
            (this->S[j]->y).row(y).at<double>(0),
            (this->S[j])->frameNumber, this->S[i]->x, this->S[i]->y, y_neg_0,
            (this->S[i])->frameNumber);

        (this->S[j]->grad).at<double>(y) -= lambda * (k_0 - k_1);

        //}
      }
    }
  }

  if (this->verbose > 1) {
    cout << "SMO step: lambda=" << lambda << '\n';
    cout << "--------------------" << endl;
  }
}

/**
 * Process old
 *
 * @return triplet for SMOstep
 */
tuple<int, cv::Mat, cv::Mat> OLaRank_old::processOld() {

  int i = rand() % this->S.size();

  cv::Mat y_plus, y_neg;
  double grad_max = -INFINITY;
  // cout<<"MAX VALUE: "<<grad_max;


  for (int y = 0; y < this->S[i]->x.rows; ++y) {
      if ((this->S[i]->beta).at<double>(y) <
        (y == this->S[i]->label) * this->parameters.C) {
      if ((this->S[i]->grad).at<double>(y) > grad_max) {
        grad_max = (this->S[i]->grad).at<double>(y);
        y_plus = (this->S[i]->y).row(y);
      }
    }
  }

  int y_neg_idx;

  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;

  minMaxLoc( this->S[i]->grad, &minVal, &maxVal, &minLoc, &maxLoc);
  y_neg_idx = minLoc.x;
  y_neg = (this->S[i]->y).row(y_neg_idx);

  if (this->verbose > 1) {
    cout << "ProcessOld step: \n"
         << "(i,y_plus,y_neg) " << i << " " << y_plus(0) << " " << y_neg(0)
         << " " << endl;
    cout << "--------------------" << endl;
  }

  tuple<int, cv::Mat, cv::Mat> result = make_tuple(i, y_plus, y_neg);
  return result;
}

/**
 * Optimize
 *
 * @return triplet for SMOstep
 */
tuple<int, cv::Mat, cv::Mat> OLaRank_old::optimize() {

  int i = rand() % this->S.size();


  while (cv::sum((abs(this->S[i]->beta)))[0] == 0) {
    i = rand() % this->S.size();
  }

  cv::Mat y_plus, y_neg;
  double grad_max = -INFINITY;
  double grad_min = INFINITY;
  // cout<<"MAX VALUE: "<<grad_max;
  // cout<<this->S[i].beta;
  for (int y = 0; y < this->S[i]->x.rows; ++y) {
    if ((this->S[i]->beta).at<double>(y) <
        this->parameters.C * (int)(y == this->S[i]->label)) {

      if ((this->S[i]->grad).at<double>(y) > grad_max) {
        grad_max = (this->S[i]->grad).at<double>(y);
        y_plus = (this->S[i]->y).row(y);
      }
    }

    if ((this->S[i]->beta).at<double>(y) != 0) {
      if ((this->S[i]->grad).at<double>(y) < grad_min) {
        grad_min = (this->S[i]->grad).at<double>(y);
        y_neg = (this->S[i]->y).row(y);
      }
    }
  }

  if (this->verbose >= 1) {
    // cout<<this->S[i].beta<<endl;
    cout << "Optimize step: \n"
         << "(i,y_plus,y_neg) " << i << " " << y_plus.at<double>(0)
         << " " << y_neg.at<double>(0)
         << " " << endl;
    cout << "--------------------" << endl;
  }

  tuple<int, cv::Mat, cv::Mat> result = make_tuple(i, y_plus, y_neg);
  return result;
}

/**
 * Predict a label for the new pattern
 *
 * @param newX - pattern to be  used for prediction
 * @return label of the pattern
 */
int OLaRank_old::predict(const cv::Mat &newX) {
  int y_hat = 0;
  int y = 0;

  double current = -INFINITY;
  double best = current;
  int bestIdx = 0;

  for (int k = 0; k < newX.rows; ++k) {
    y = k;
    current = 0;

    for (int i = 0; i < this->S.size(); ++i) {
      for (int yhat = 0; yhat < this->S[i]->x.rows; ++yhat) {
        y_hat = yhat;

        if ((this->S[i]->beta).at<double>(yhat) != 0) {
          // the below has to be multiplied by the velocities kernel
          current +=
              (this->S[i]->beta).at<double>(yhat) *
              this->calculate_kernel(newX, y, this->S[i]->x, y_hat);
        }
      }
    }

    if (current >= best) {
      bestIdx = k;
      best = current;
    }
  }
  return bestIdx;
}

/**
 *  Loss function
 * @param y
 * @param y_hat
 * @return value of the loss function
 */
double OLaRank_old::loss(const cv::Mat &y, const cv::Mat &y_hat) {
  // find intersection of the two rectangles
  double a1 = y.at<double>(1);
  double b1 = y.at<double>(2);
  double a2 = a1 + y.at<double>(3);
  double b2 = b1 + y.at<double>(4);

  double c1 = y_hat.at<double>(1);
  double d1 = y_hat.at<double>(2);
  double c2 = c1 + y_hat.at<double>(3);
  double d2 = d1 + y_hat.at<double>(4);

  double intersection =
      std::max(std::min(a2, c2) - std::max(a1, c1), double(0)) *
      (std::max(std::min(b2, d2) - std::max(b1, d1), double(0)));
  double loss =
      1 - intersection / (y.at<double>(3) * y.at<double>(4) + y_hat.at<double>(3) * y_hat.at<double>(4) - intersection);

  return loss;
}

/**
 * Calculates objective function for structured output SVM.
 * @return objective value
 */
double OLaRank_old::calculateObjective() {

  double objective = 0;
  double s = 0;
  for (int i = 0; i < this->S.size(); ++i) {
    for (int k = 0; k < this->S[i]->x.rows; ++k) {
      objective += loss((this->S[i]->y).row(this->S[i]->label),
                        (this->S[i]->y).row(k)) *
                   (this->S[i]->beta).at<double>(k);
    }

    for (int j = i; j < this->S.size(); ++j) {
      for (int y = 0; y < this->S[i]->x.rows; ++y) {
        for (int yhat = 0; yhat < this->S[j]->x.rows; ++yhat) {
            /*
          s = ((S[i]->beta).at<double>(y) *
               (S[j]->beta).at<double>(yhat) * kernel_fast(S[i]->x,
                                                           (S[i]->y).row(y),
                                                           S[i]->frameNumber,
                                                           S[j]->x,
                                                           (S[i]->y).row(yhat),
                                                           S[j]->frameNumber));
            */
          if (i == j) {
            s = s * 0.5;
          }

          objective -= s;
        }
      }
    }
  }

  return objective;
}

double OLaRank_old::kernel_fast(const cv::Mat &x, const cv::Mat &y_loc, int y,
                                int frameNumber_1, const cv::Mat &xp,
                                const cv::Mat &yp_loc, int yp,
                                int frameNumber_2) {
  // to save only half of the kernels
  if (frameNumber_2 < frameNumber_1) {
    return kernel_fast(xp, yp_loc, yp, frameNumber_2, x, y_loc, y,
                       frameNumber_1);
  }

  // we can always assume that frameNumber_1>=frameNumber_2
  Key key(frameNumber_1, frameNumber_2);

  auto it = (this->kern).find(key);

  double result = 0;

  if (it == (this->kern).end()) {
    // key is not found

    // allocate memory for the new matrix
      cv::Mat kern_matrix = cv::Mat::ones(x.rows, xp.rows, CV_64F);

    // set all elements to -infinity which means that the kernel value wasn't
    // calculated
    kern_matrix = (DBL_MIN) * (kern_matrix);

    // calculate the value

    result = calculate_kernel(x, y, xp, yp);

    (kern_matrix).at<double>(y, yp) = result;
    // add it to the kernel map
    (this->kern).insert({key, kern_matrix});

  } else {
    // key is found - check if value is calculated

      if ((it->second).at<double>(y, yp) != DBL_MIN) {
      // the value was previously calculated - return it
      result = (it->second).at<double>(y, yp);
    } else {
      // the value wasn't calculated. Firstly, calculate it and store in the
      // kernel

      result = calculate_kernel(x, y, xp, yp);
      (it->second).at<double>(y, yp) = result;
      // check if it will change
    }
  }

  return result;
}

double OLaRank_old::calculate_kernel(const cv::Mat &x, int y,
                                     const cv::Mat &xp, int yp) {
  return this->svm_kernel->calculate(x, y, xp, yp);
}

/**
 *  Create an instance using given set of parameters
 */
OLaRank_old::OLaRank_old(Kernel *svm_kernel_, params &learningParams,
                         int &balance, int &m_, int &verbose_) {

  svm_kernel = svm_kernel_;
  parameters = learningParams;

  m = m_;
  B = balance;
  verbose = verbose_;

  // this->kern=new unordered_map<Key,arma::mat*,KeyHash, KeyEqual>;
}

/**
 * Predict label for the new pattern, newX, and process it.
 *
 * @param newX 				- 		pattern
 * @return 					- 		predicted label for the
 * pattern
 */
int OLaRank_old::processAndPredict(const cv::Mat &newX, const cv::Mat &newY,
                                   int frameNumber) {

  int y_hat_idx = this->predict(newX);

  // mat y_hat=newY.row(y_hat_idx);

  tuple<cv::Mat, cv::Mat, cv::Mat> p_new =
      this->processNew(newX, newY, y_hat_idx, frameNumber);

  cv::Mat y_plus, y_neg;
  cv::Mat grad = cv::Mat::zeros(1, newX.rows, CV_64F);
  tie(y_plus, y_neg, grad) = p_new;

  // add new element into set S
  supportData *support = new supportData(newX, newY, y_hat_idx, 0,
                                         newX.rows, frameNumber);
  (support->grad) = grad;

  double i = this->S.size();
  this->S.push_back(support);

  smoStep(i, y_plus, y_neg);
  // this->checkIfConstraintsSatisfied();

  budgetMaintance();

  for (int ii = 0; ii < this->parameters.n_R; ++ii) {

    if (this->S.size() != 0) {

      tuple<double, cv::Mat, cv::Mat> p_old = this->processOld();
      tie(i, y_plus, y_neg) = p_old;

      smoStep(i, y_plus, y_neg);

      // this->checkIfConstraintsSatisfied();

      budgetMaintance();
    }

    for (int j = 0; j < this->parameters.n_O; ++j) {
      if (this->S.size() != 0) {
        tuple<double, cv::Mat, cv::Mat> optimize = this->optimize();

        tie(i, y_plus, y_neg) = optimize;
        smoStep(i, y_plus, y_neg);
      }
      // this->checkIfConstraintsSatisfied();
    }
  }

  if (this->verbose > 2) {

    cout << "---------------------------------------\n";
    cout << "OptValue " << this->calculateObjective() << "\n";
    cout << "---------------------------------------\n";
  }

  return y_hat_idx;
}

/**
 * Process (input,output) pair
 *
 * @param newX 					- pattern
 * @param y_hat					- label
 */
void OLaRank_old::process(const cv::Mat &newX, const  cv::Mat &y_hat,
                          int y_hat_label,
                          int frameNumber) {

  tuple<cv::Mat, cv::Mat, cv::Mat> p_new =
      this->processNew(newX, y_hat, y_hat_label, frameNumber);

  cv::Mat y_plus, y_neg;
  cv::Mat grad = cv::Mat::zeros(1, newX.rows, CV_64F);
  tie(y_plus, y_neg, grad) = p_new;

  // add new element into set S
  supportData *support = new supportData(newX, y_hat, y_hat_label, 0,
                                         newX.rows, frameNumber);
  (support->grad) = grad;

  // delete support;

  double i = this->S.size();
  this->S.push_back(support);

  /*

   debug code

   */
  //    int test=0;
  //
  //    for (int i=0; i<this->S.size(); ++i) {
  //
  //        int sum=0;
  //
  //        for (int j=0; j< this->K; ++j) {
  //            //if (abs((*this->S[i]->beta)(j))>=1.0e-7) {
  //
  //            if(abs((*this->S[i]->beta)(j))!=0){
  //                test+=1;
  //            }else{
  //                sum++;
  //            }
  //        }
  //    }
  //
  //    std::cout<<"Number of  support Vectors :
  //    "<<test<<"/"<<this->S.size()<<std::endl;
  //
  //
  //    std::vector<Key> keys;
  //    keys.reserve(this->kern->size());
  //
  //
  //    for(auto kv : *this->kern) {
  //        keys.push_back(kv.first);
  //    }
  //
  //    std::cout<<"Cache size: "<<keys.size()<<"/"
  //    <<(this->S.size()*(this->S.size()+1))/2<<std::endl;

  smoStep(i, y_plus, y_neg);
  // this->checkIfConstraintsSatisfied();

  budgetMaintance();

  for (int ii = 0; ii < this->parameters.n_R; ++ii) {

    tuple<double, cv::Mat, cv::Mat> p_old = this->processOld();
    tie(i, y_plus, y_neg) = p_old;

    smoStep(i, y_plus, y_neg);

    // this->checkIfConstraintsSatisfied();

    budgetMaintance();

    for (int j = 0; j < this->parameters.n_O; ++j) {
      tuple<double, cv::Mat, cv::Mat> optimize = this->optimize();

      tie(i, y_plus, y_neg) = optimize;
      smoStep(i, y_plus, y_neg);

      // this->checkIfConstraintsSatisfied();
    }
  }

  if (this->verbose > 0) {

    cout << "---------------------------------------\n";
    cout << "OptValue " << this->calculateObjective() << "\n";
    cout << "---------------------------------------\n";
  }
}

void OLaRank_old::testIfObjectiveIncreases() {

  int i;
  cv::Mat y_plus, y_neg;

  for (int ii = 0; ii < this->parameters.n_R; ++ii) {
    for (int j = 0; j < 1; ++j) {
      tuple<double, cv::Mat, cv::Mat> optimize = this->optimize();

      tie(i, y_plus, y_neg) = optimize;

      cout << "Beta (should be non zero): " << this->S[i]->beta << "\n";
      cout << "Label for the current pattern: " << this->S[i]->label << "\n";
      smoStep(i, y_plus, y_neg);
      cout << "OptValueCHECK " << this->calculateObjective() << "\n";
    }
  }
}

/**
 *  Initialize the first training pattern and label
 *
 *  @param x  			first pattern
 *  @param label  		label of the first pattern
 *  @param y 			first label
 */
void OLaRank_old::initialize(const cv::Mat &x, const cv::Mat &y, const int label,
                             int frameNumber) {

  supportData *s1 = new supportData(x, y, label, m, x.rows, frameNumber);
  S.push_back(s1);

  this->process(x, y, label, frameNumber);
}

double OLaRank_old::recomputeGradient(int i, int y) {

  double grad = 0;
  // update gradient g_i(y)
  grad -= this->loss((this->S[i]->y).row(y),
                     (this->S[i]->y).row(this->S[i]->label));

  for (int j = 0; j < this->S.size(); ++j) {
    for (int yhat = 0; yhat < this->S[i]->x.rows; ++yhat) {
      grad -= (this->S[j]->beta).at<double>(yhat) *
              this->kernel_fast(
                  this->S[i]->x, this->S[i]->y,
                  (this->S[i]->y).row(y).at<double>(0),
                  (this->S[i])->frameNumber, this->S[j]->x, this->S[j]->y,
                  (this->S[j]->y).row(yhat).at<double>(0),
                  (this->S[j])->frameNumber);
    }
  }
  return grad;
}

void OLaRank_old::deleteKernelValues(int frameNumber) {

  int f1 = 0;
  int f2 = 0;

  for (int i = 0; i < this->S.size(); i++) {
    int frameNumber_2 = this->S[i]->frameNumber;

    if (frameNumber <= frameNumber_2) {
      f1 = frameNumber;
      f2 = frameNumber_2;
    } else {
      f2 = frameNumber;
      f1 = frameNumber_2;
    }

    // get the key
    Key key(f1, f2);

    auto it = this->kern.find(key);

    if (it != this->kern.end()) {
      // if the key is present. Delete the matrix associated with it from the
      // heap

      // and then from the unordered map
      this->kern.erase(key);
    }
  }
}

std::ostream &operator<<(std::ostream &strm, const OLaRank_old &s) {

  strm << "OLaRank parameters: \n";
  strm << "C                 : " << s.parameters.C << "\n";
  strm << "n_R               : " << s.parameters.n_R << "\n";
  strm << "n_O               : " << s.parameters.n_O << "\n";
  strm << "B                 : " << s.B << "\n";
  strm << "Kernel: \n" << s.svm_kernel->getInfo() << "\n";

  return strm;
}
OLaRank_old::~OLaRank_old() {

  {

    // Delete every memory associated with all the matrices
    for (int i = 0; i < this->S.size(); i++) {
      deleteKernelValues(i);
    }

    for (int j = 0; j < this->S.size(); ++j) {
      supportData *s = S[j];
      delete s;
    }

    this->S.clear();

    this->clear();
    delete svm_kernel;
  }
}
