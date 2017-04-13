#ifndef TRAINSVM_H_
#define TRAINSVM_H_

#include <cv.h>
#include <highgui.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>
#include <sstream>
#include "stdafx.h"
#include <opencv2\opencv.hpp>


class TrainSVM 
{
public:
    TrainSVM();
    virtual ~TrainSVM();
	void trainDetector(void);
	void trainClassifier(void);
	void getLBPFeatures(const cv::Mat& image, cv::Mat& features);
	void getHOGFeatures(const cv::Mat& image, cv::Mat& features);
	cv::Ptr<cv::ml::TrainData> getClassiferSet(void);
	cv::Ptr<cv::ml::TrainData> getDetectorSet(void);
private:
	cv::HOGDescriptor *hog;
};

#endif
/* TRAINSVM_H_ */