//
//  LinearKernel.h
//  Robust_tracking_by_detection
//
//  Created by Ivan Bogun on 1/27/15.
//
//

#ifndef __arma_Robust_tracking_by_detection__LinearKernel__
#define __arma_Robust_tracking_by_detection__LinearKernel__

#include <stdio.h>
#include "arma_Kernel.h"

namespace armadillo {
class LinearKernel:public armadillo::Kernel {
public:
    void preprocess(std::vector<armadillo::supportData*>& S,int B){};
    double calculate(arma::mat& x1, int r1, arma::mat& x2, int r2){
        return arma::dot(x1.row(r1),x2.row(r2));
    };
    std::string getInfo(){
        return "Linear kernel";
    }

    ~LinearKernel(){
    }
};

}

#endif /* defined(__Robust_tracking_by_detection__LinearKernel__) */
