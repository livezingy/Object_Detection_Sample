/* --------------------------------------------------------
* author：livezingy
*
* BLOG：http://www.livezingy.com
*
* Development Environment：
*      Visual Studio V2013
*      opencv3.1
*      Tesseract3.04
*
* Version：
*      V1.0    20170414

--------------------------------------------------------- */
#include <cv.h>
#include <highgui.h>
#include <ml.h>
#include <opencv2\opencv.hpp>
#include "TrainSVM.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>
#include <time.h>
#include "util.h"

using namespace cv;
using namespace cv::ml;
using namespace std;

HOGDescriptor *hogDetect;
Ptr<cv::ml::SVM> svmClassifier;
vector<float> detector;
string svmClassiferPath = "etc/LBP_RBF_classifer.xml";
string svmDetectPath = "etc/HOG_SVM.txt";
string descriptorType = "LBP";
string videoFilename = "etc/Sample.mp4";
TrainSVM trainSVM;

Mat VerticalProjection(const Mat& src);

void detectMulti(void);

int frameH;
int frameW;

//传送带正常运转时的步长
int conCount = 0;
int truckNum = 0;
int carNum = 0;

int main(int argc, char** argv)
{
	ifstream fileIn(svmDetectPath, ios::in);
	float val = 0.0f;
	if (!fileIn.is_open())
	{
		fprintf(stdout, ">> No detector file, training according the imageset in etc/TrainDetector...\n");
		trainSVM.trainDetector();
	}

	while (!fileIn.eof())
	{
		fileIn >> val;
		detector.push_back(val);
	}
	fileIn.close();
	
	//必须设置的和分类器的参数相同
	hogDetect = new HOGDescriptor(cvSize(198, 214), cvSize(38, 54), cvSize(8, 8), cvSize(19, 27), 3);
	hogDetect->setSVMDetector(detector);// 使用自己训练的检测器进行目标检测

	FileStorage fs(svmClassiferPath, FileStorage::READ);
	if (!(fs.isOpened()))
	{
		fprintf(stdout, ">> No Classifier file, training according the imageset in etc/TrainClassifier...\n");
		trainSVM.trainClassifier();
	}
	svmClassifier = cv::ml::SVM::create();
	svmClassifier = cv::ml::StatModel::load<cv::ml::SVM>(svmClassiferPath);

	detectMulti();
	
	getchar();
	
	return 0;
}



void detectMulti(void)
{
	VideoCapture capture(videoFilename);

	if (!capture.isOpened())
	{
		//error in opening the video input
		cerr << "Unable to open video file, Please check the name of vedio and try again." << videoFilename << endl;
		//exit(EXIT_FAILURE);
	}

	frameH = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	frameW = capture.get(CV_CAP_PROP_FRAME_WIDTH);

	// resize input image to (128,64) for compute
	Size dsize = Size(320, 352);//198，214
	Mat trainImg = Mat(dsize, CV_32S);
	
	int keyboard = 0;
	int num = 0;

	Mat frame, grayFrame, testImage, preImage, FrameK, FrameK1, FrameDiff;

	Mat feature;
	
	int equDiff;

	//long frameToStart = 30;
	//capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	//cout << "从第" << frameToStart << "帧开始读" << endl;

	Rect ObjRect(frameW / 2, 0, frameW / 2, frameH);

	int preResult = 0;

	int count = 0;

	while (((char)keyboard != 'q') && ((char)keyboard != 27))
	{
		double t = (double)getTickCount();
		//read the current frame
		if (!capture.read(frame))
		{
			cerr << "Unable to read next frame." << endl;
			cerr << "Exiting..." << endl;
			break;
		}

		cvtColor(frame, grayFrame, CV_BGR2GRAY);

		grayFrame(ObjRect).copyTo(testImage);
		
		FrameK = VerticalProjection(testImage);
		
		if(num > 2)
		{	
			absdiff(FrameK, FrameK1, FrameDiff);

			int* p_dst = FrameDiff.ptr<int>(0);

			int tmpSum = 0;
			for (int i = 0; i < frameW/2; i++)
			{
				tmpSum += p_dst[i];
			}
			equDiff = cvRound(tmpSum / frameW);
			
			if(equDiff > 500)
			{
				vector<Rect> found, found_filtered;
		
				hogDetect->detectMultiScale(testImage, found, 0, Size(64, 64), Size(0, 0), 1.1, 2);
		
				size_t i, j;

				//去掉空间中具有内外包含关系的区域，保留大的
				for (i = 0; i < found.size(); i++)
				{
					Rect r = found[i];
					for (j = 0; j < found.size(); j++)
					{
						if (j != i && (r & found[j]) == r)
							break;
					}
					if (j == found.size())
						found_filtered.push_back(r);

				}

				if (found_filtered.size() > 0)
				{								
					cv::Point pTl(found_filtered[0].tl().x + frameW / 2, found_filtered[0].tl().y);
					cv::Point pBR(found_filtered[0].br().x + frameW / 2, found_filtered[0].br().y);
					
					rectangle(frame, pTl, pBR, cv::Scalar(255, 0, 255), 3);

					cv::rectangle(frame, cv::Point(10, 2), cv::Point(400, 20),
						cv::Scalar(255, 255, 255), -1);

					if ((0 == preResult))
					{
						Rect reRect(found_filtered[0].tl().x, 0, found_filtered[0].br().x - found_filtered[0].tl().x, frameH);

						testImage(reRect).copyTo(preImage);
		
						resize(preImage, trainImg, dsize);
						
						trainSVM.getLBPFeatures(preImage, feature);					

						preResult = ((int)svmClassifier->predict(feature));

						if (preResult != 0)
						{
							count = 0;
							
							if (1 == preResult)
							{
								truckNum++;
							}
							else
							{
								carNum++;
							}
						}
					}	
			    }
				else
				{
					if (count > 10)
					{
						preResult = 0;
						count = 0;
					}
					
				}
			}
		}
		else
		{
			num++;
		}

		count++;
		
		FrameK.copyTo(FrameK1);

		char str[10];

		std::sprintf(str, "%d", truckNum);

		string tmpTruck = str;

		std::sprintf(str, "%d", carNum);
		string tmpCar = str;
		
		//std::sprintf(str, "%d", equDiff);
		//string tmpDiff = str;

		cv::rectangle(frame, cv::Point(10, 2), cv::Point(300, 20),
			cv::Scalar(255, 255, 255), -1);

		string tmpStr = "Truck tyres: " + tmpTruck + ";  " + "Car tyres: " + tmpCar + ";  ";//+ "equDiff: " + tmpDiff + ";";

		//string frameNumberString = ss.str();
		putText(frame, tmpStr, cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			
		t = (double)getTickCount() - t;
		printf("detection time = %gms\n", t*1000. / cv::getTickFrequency());

		imshow("Frame", frame);

		keyboard = waitKey(3);
	}
	capture.release();

	printf("There are %d truck tyres and %d car tyres\n", truckNum, carNum);

	printf("Press any key to exit");

	cv::destroyAllWindows();

	
}


Mat VerticalProjection(const Mat& src)//, Mat& dst)
{
	// accept only char type matrices  
	CV_Assert(src.depth() != sizeof(uchar));
	Mat dst;
	dst.create(1, src.cols, CV_32F);

	int i, j;
	const uchar* p;
	int* p_dst = dst.ptr<int>(0);

	for (j = 0; j < src.cols; j++)
	{
		p_dst[j] = 0;
		for (i = 0; i < src.rows; i++)
		{
			p = src.ptr<uchar>(i);
			p_dst[j] += p[j];

			int tmpP = p[j];
			int tmpPdst = p_dst[j];
		}
	}

	return dst;//VerProjectionImage(dst);
}