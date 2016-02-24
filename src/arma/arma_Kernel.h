//
//  Kernel.h
//  Robust Struck
//
//  Created by Ivan Bogun on 10/5/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef __arma_Robust_Struck__Kernel__
#define __arma_Robust_Struck__Kernel__
#include <opencv2/opencv.hpp>
#include "armadillo"
#include "arma_supportData.h"

namespace armadillo {
class Kernel {

public:

    virtual ~Kernel(){};
    virtual void preprocess(std::vector<armadillo::supportData*>& S,int B)=0;
    virtual double calculate(arma::mat& x,int r1,arma::mat& x2,int r2)=0;
    virtual std::string getInfo()=0;
    virtual arma::rowvec predictAll(arma::mat& newX,std::vector<armadillo::supportData*>& S, int B);
};

} // armadillo namespace
#endif /* defined(__Robust_Struck__Kernel__) */
