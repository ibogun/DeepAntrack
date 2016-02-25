//
//  OLaRank_old.h
//  Structured_BING
//
//  Created by Ivan Bogun on 7/17/14.
//  Copyright (c) 2014 Ivan Bogun. All rights reserved.
//

#ifndef OLARANK_OLD_H_
#define OLARANK_OLD_H_
#include <vector>

#include <tuple>

#include "../Kernels/Kernel.h"
#include <unordered_map>
#include "supportData.h"
struct params {

	double C; // C parameter in SVM notation
	int n_R;  // number of outer iterations
	int n_O;  // number of inner iterations

};

using namespace std;


struct Key {
    int frameNumber_1;
    int frameNumber_2;

    Key(int frameNumber_1_,int frameNumber_2_){
        this->frameNumber_1=frameNumber_1_;
        this->frameNumber_2=frameNumber_2_;
    }
};

struct KeyHash {
    std::size_t operator()(const Key& k) const
    {
        return std::hash<int>()(k.frameNumber_1*k.frameNumber_2) ;
    }
};

struct KeyEqual {
    bool operator()(const Key& lhs, const Key& rhs) const
    {
        return lhs.frameNumber_1 == rhs.frameNumber_1 && lhs.frameNumber_2 == rhs.frameNumber_2;
    }
};





class OLaRank_old {

private:

    params parameters;
    Kernel* svm_kernel;
    int verbose;
    // frameNumber - list of <frameNumber,value> pairs
    unordered_map<Key,cv::Mat,KeyHash, KeyEqual> kern;
    friend std::ostream& operator<<(std::ostream&, const OLaRank_old&);

public:
    // size of the pattern
    int B;
    int m;
    vector<supportData*> S;
    std::vector<std::pair<double, double>> velocity;

    // populate this first
    // frameNumber - location unordered map
    unordered_map<int,std::pair<double, double>> locations;

    void clear(){
        this->kern.clear();
        this->velocity.clear();
        this->locations.clear();
    };
    OLaRank_old(Kernel*,int seed=1);
    OLaRank_old(Kernel*,params&, int&, int&,int&);

    void setParameters(params&, int&, int&, int&);

    void initialize(const cv::Mat&, const cv::Mat&,const int,int);
    int processAndPredict(const cv::Mat&,const cv::Mat&,int);
    void process(const cv::Mat&,const cv::Mat&,const int,int);

    double kernel_fast(const cv::Mat&,const cv::Mat&,int,int,const cv::Mat&,const cv::Mat&,int,int);
    double calculate_kernel(const cv::Mat&,int,const cv::Mat&,int);

    double calculateObjective();
    double loss(const cv::Mat&, const cv::Mat&);

    tuple<cv::Mat, cv::Mat, cv::Mat> processNew(const cv::Mat&, const cv::Mat&,
                                                int,int);
    int budgetMaintance();

    void smoStep( int, const cv::Mat&, const cv::Mat&);

    void testIfObjectiveIncreases();
    tuple<int, cv::Mat, cv::Mat> processOld();
    tuple<int, cv::Mat, cv::Mat> optimize();

    void checkIfConstraintsSatisfied();

    int predict(const cv::Mat&);

    std::vector<double> predictAll(const cv::Mat&);
    void learn(const cv::Mat&,const cv::Mat&);

    double recomputeGradient(int i,int y);
    void deleteKernelValues(int frameNumber);

    ~OLaRank_old();
};


#endif /* STRUCK_H_ */
