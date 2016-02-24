//
//  Kernel.cpp
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#include "Kernel.h"
#include <vector>

std::vector<double> Kernel::predictAll(const cv::Mat &newX,
                                       std::vector<supportData *> &S, int B) {
  // preprocess first
  preprocess(S, B);
  int n = newX.rows;
  double y_hat = 0;
  double y = 0;

  std::vector<double> scores(n);
  for (int k = 0; k < newX.rows; ++k) {
    y = k;
    double current = 0;
    for (int i = 0; i < S.size(); ++i) {
      for (int yhat = 0; yhat < S[i]->x.rows; ++yhat) {
        y_hat = yhat;
        double beta = S[i]->beta.at<double>(0, yhat);
        if (beta != 0) {
          current +=
            beta * calculate(newX, y, S[i]->x, y_hat);
        }
      }
    }
    scores[k] = current;
  }
  return scores;
}
