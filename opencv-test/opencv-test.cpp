
/* Saliency testing using OpenCV

A. Erdem and E. Erdem. (2003): Visual saliency estimation by nonlinearly integrating features using region covariances
http://www.journalofvision.org/content/13/4/11.full.pdf

B. Celikkale, A. Erdem and E. Erdem (2013): Visual Attention-driven Spatial Pooling for Image Memorability.
IEEE Computer Vision and Pattern Recognition Workshops (CVPRW), Portland, Oregon, USA, June 2013.

Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(11), 1254–1259.

@misc{mit-saliency-benchmark,
  author       = {Zoya Bylinskii and Tilke Judd and Fr{\'e}do Durand and Aude Oliva and Antonio Torralba},
  title        = {MIT Saliency Benchmark},
  howpublished = {http://saliency.mit.edu/}
}
*/

/*
Wavelet decomposition seems relevant
Saliency Map feeds into SDR
Gabor or feature filtering?

1) Focus of Attention is shifted to the location of the winner neuron
2) The global inhibition of the WTA (winner-takes-all) is triggered and completely inhibits (resets) all WTA neurons
3) Local inhibition is transiently activated in the Saliency Map, in an area with the size and new location of the FOA

3 prevents FOA fromimmediately returning to a previously attended location. In order to slightly bias the model to subsequently jump to salient locations spatially close to the currently attended location, a small excitation is transiently activated in the SM, in a new surroundof the FOA ("proxminity preference" rule of Kock and Ullman)

With no top-down instruction the FOA is a disk radius fixed to one sixth of width/height
FOA jumps from one salient location to the next in approx. 30-70ms
An attended area is inhibited for approx. 500-900ms

*/

#include <opencv2/opencv.hpp>

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include <time.h>

#include "opencv2/bioinspired/retina.hpp"
#include "opencv2/face.hpp"
#include "BMS.h"

using namespace std;
using namespace cv;

#ifndef uint8_t
#define uint8_t		unsigned char
#define uint16_t	unsigned short
#define int32_t		int
#endif

// Max number of potential saccade candidate
static const vector<cv::Vec3f>::size_type cMaxNumOfSaccadeCandidates = 4;

int getOtsuThreshold(cv::Mat& src, float *maxValue);
int connectedComponents(cv::Mat &L, const cv::Mat &I, int connectivity);

cv::MatND retinaHistogram_magno;
int histogramSize = 256;
 

int main( int argc, char** argv )
{
	bool bCalc_Regions = true;
	bool bCalc_HoughCircles = false;
	bool bCalc_Histogram = true;
	bool bCalc_BMS = false;

    // Declare the retina input buffer... that will be fed differently in regard of the input media
    cv::Mat inputFrame;

	cv::VideoCapture videoCapture; // in case a video media is used, its manager is declared here

	int frameCount = 0;
	time_t start, end;
	double fps, sec;

	double windowWidthScale  = 0.33;
	double windowHeightScale = 0.33;
 
	videoCapture.open("./Wildlife.wmv");
	//videoCapture.open(0);

	time(&start);

	videoCapture >> inputFrame;
	frameCount++;
	if (inputFrame.empty())
	{
		cv::String fileName = "./Burt.jpg";
		inputFrame = cv::imread(fileName, 0);

		windowWidthScale = 0.125;
		windowHeightScale = 0.125;
	}

	if (!inputFrame.empty())
	    resize(inputFrame, inputFrame, cv::Size(), windowWidthScale, windowHeightScale, INTER_AREA);
 
    // Declare retina output buffers
    cv::Mat retinaOutput_parvo, prev_retinaOutput_parvo;
    cv::Mat retinaOutput_magno, prev_retinaOutput_magno;
 
    cv::Mat retinaThreshold_magno = cv::Mat::zeros(inputFrame.rows, inputFrame.cols, CV_32F);

	cv::Mat retinaAreaOfInterest = cv::Mat::zeros(retinaThreshold_magno.rows, retinaThreshold_magno.cols, CV_8UC3);
	cv::Mat	retinaAOIgreyscale(retinaAreaOfInterest.rows, retinaAreaOfInterest.cols, CV_8UC1);
	cv::Mat retinaAOIbinary(retinaAreaOfInterest.rows, retinaAreaOfInterest.cols, CV_8UC1);

	int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
	bool useLogSampling = false;
 
    //////////////////////////////////////////////////////////////////////////////
    // Program start in a try/catch safety context (Retina may throw errors)
    try
    {
        // create a retina instance with default parameters setup, uncomment the initialisation you wanna test
		// http://docs.opencv.org/doc/tutorials/contrib/retina_model/retina_model.html
        cv::Ptr<bioinspired::Retina> myRetina;

		// Activate log sampling? (favour foveal vision and subsamples peripheral vision)
		if (useLogSampling)
		{
			myRetina = cv::bioinspired::createRetina(inputFrame.size(), true, cv::bioinspired::RETINA_COLOR_BAYER, true, 2.0, 10.0);
		}
		else
			// -> else allocate "classical" retina :
			myRetina = cv::bioinspired::createRetina(inputFrame.size());

        // save default retina parameters file in order to let you see this and maybe modify it and reload using method "setup"
        myRetina->write("RetinaDefaultParameters.xml");
 
        // load parameters if file exists
        myRetina->setup("RetinaSpecificParameters.xml");
        myRetina->clearBuffers();
 
		myRetina->run(inputFrame);
		myRetina->getParvo(retinaOutput_parvo);
		myRetina->getMagno(retinaOutput_magno);
                
		retinaOutput_parvo.copyTo(retinaAreaOfInterest);

		// Setup a random color palette for all the labels
		std::vector<cv::Vec3b> colors;
		colors.push_back(cv::Vec3b(0, 0, 0));//background
		for(int label = 1; label <= cMaxNumOfSaccadeCandidates; ++label){
			colors.push_back(cv::Vec3b( label*64, (2*label)*32, (3*label)*32 ));
		}

        // processing loop with stop condition
        //while(frameCount < 50)
		while(videoCapture.grab())
        {
			// If using video stream, then, grabbing a new frame
             
            if (videoCapture.isOpened())
            {
                videoCapture >> inputFrame;
				frameCount++;

				if (!inputFrame.empty())
				    resize(inputFrame, inputFrame, cv::Size(), windowWidthScale, windowHeightScale, INTER_AREA);

	            prev_retinaOutput_parvo = retinaOutput_parvo;
	            prev_retinaOutput_magno = retinaOutput_magno;
			}
 
            // Run retina filter
             
            myRetina->run(inputFrame);
 
            // Retrieve and display retina output
             
            // -> foveal color vision details channel with luminance and noise correction
            myRetina->getParvo(retinaOutput_parvo);
 
            // -> peripheral monochrome motion and events (transient information) channel
            myRetina->getMagno(retinaOutput_magno);
 
            float maxMagnoHistVal = 0.f;
            int Magno_OtsuThreshold = getOtsuThreshold(retinaOutput_magno, &maxMagnoHistVal) + 16;
 
			//if (Magno_OtsuThreshold < 128)
            {
				// Create the Threshold version of the magno pathway image
                double high_thres = cv::threshold(retinaOutput_magno, retinaThreshold_magno, 
					(double)Magno_OtsuThreshold, 255.0, THRESH_TOZERO);

				// Copy parvo pathway details using the thresholded magno pathway as a mask
				retinaAreaOfInterest = cv::Scalar(0);
                retinaOutput_parvo.copyTo(retinaAreaOfInterest, retinaThreshold_magno);

				// To grey scale we go
				//retinaAreaOfInterest.convertTo(retinaAOIgreyscale, CV_8UC1);

				cv::Mat labelImage(inputFrame.size(), CV_32S);
				cv::Mat labelFrame = cv::Mat::zeros(labelImage.rows,labelImage.cols,inputFrame.type());
		
				if (bCalc_Regions)
				{
					int nLabels = connectedComponents(labelImage, retinaThreshold_magno, 8);

					// Paint the label frame
					for(int r = 0; r < labelFrame.rows; ++r){
						for(int c = 0; c < labelFrame.cols; ++c){
							int label = labelImage.at<int>(r, c);
							cv::Vec3b &pixel = labelFrame.at<cv::Vec3b>(r, c);
							pixel = colors[label&(cMaxNumOfSaccadeCandidates-1)];
						}
					}

					// Apply a canny edge filter to the label regions
					cv::Mat retinaAOIedges = cv::Mat::zeros(retinaAreaOfInterest.rows, retinaAreaOfInterest.cols, CV_8UC1);
					Canny(labelFrame, retinaAOIedges, 
						0.5*high_thres, (double)Magno_OtsuThreshold);//high_thres); // 0.66*mean,1.33*mean

					// Threshold the edges image to binary
					retinaAOIbinary = cv::Mat::zeros(retinaAOIedges.rows, retinaAOIedges.cols, CV_8UC1);
					cv::threshold(retinaAOIedges, retinaAOIbinary, 0.0, 255.0, THRESH_BINARY);
					//cv::imshow("retinaAOIBinary", retinaAOIbinary);

					// Find contours on the binary image
					vector<vector<cv::Point>> contours;
					vector<cv::Vec4i> hierarchy;
					//cv::findContours(retinaAOIbinary, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

					// If any found we can draw them back onto the label frame
					if( !contours.empty() && !hierarchy.empty() )
					{
						// iterate through all the top-level contours,
						// draw each connected component with its own random color
						int idx = 0;
						for(int label = 255; idx >= 0; idx = hierarchy[idx][0], label-- )
						{
							cv::Scalar color( label*16, label*16, label*16 );
							drawContours(labelFrame, contours, idx, color, FILLED, 8, hierarchy);
						}
					}

				}

				if (bCalc_HoughCircles)
				{
					vector<cv::Vec3f> circles;
					cv::HoughCircles(retinaThreshold_magno, circles, HOUGH_GRADIENT, 
						2, retinaThreshold_magno.rows/4, high_thres == 0 ? 100 : high_thres, 16, 
						retinaThreshold_magno.rows/8, retinaThreshold_magno.rows/6 );
				//	cv::HoughCircles(retinaAOIbinary, circles, HOUGH_GRADIENT, 
				//		2, retinaAOIbinary.rows/4, high_thres == 0 ? 100 : high_thres, 64 );
				
					for( size_t i = 0; i < circles.size() && i < cMaxNumOfSaccadeCandidates; i++ )
					{
						cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
						int radius = cvRound(circles[i][2]);
					
						// draw the circle center
						cv::circle(retinaOutput_parvo, center, 3, cv::Scalar(i*16,255,i*16), -1, 8, 0);
						cv::circle(inputFrame, center, 3, cv::Scalar(i*16,255,i*16), -1, 8, 0);
					
						// draw the circle outline
						cv::circle(retinaOutput_parvo, center, radius, cv::Scalar(i*16,i*16,255), 2, 8, 0);
						cv::circle(labelFrame, center, radius, cv::Scalar(i*16,i*16,255), 2, 8, 0);
					}
				}

				cv::Mat imageHistogram_magno = cv::Mat::ones(256, 256, CV_8U)*255;
				if (bCalc_Histogram)
				{
					cv::Mat normHist(retinaHistogram_magno);
					normalize(retinaHistogram_magno, normHist, 0, imageHistogram_magno.rows, NORM_MINMAX, CV_32F);
 
					int binWidth = cvRound((double)imageHistogram_magno.cols/histogramSize);
					for (int s = 0; s < histogramSize; s++ )
						rectangle( imageHistogram_magno, 
									cv::Point(s*binWidth,     imageHistogram_magno.rows),
									cv::Point((s+1)*binWidth, imageHistogram_magno.rows - cvRound(normHist.at<float>(s))),
									cv::Scalar::all(0), 
									-1, 8, 0 );
 
					// Plot on the histogram image a vertical two-tone rectangle at [t, t+1]
					rectangle( imageHistogram_magno,
								cv::Point(Magno_OtsuThreshold*binWidth,     imageHistogram_magno.rows),
								cv::Point((Magno_OtsuThreshold+1)*binWidth, imageHistogram_magno.rows - cvRound(normHist.at<float>(Magno_OtsuThreshold))),
								cv::Scalar::all(255),
								-1, 8, 0 );
					rectangle( imageHistogram_magno,
								cv::Point(Magno_OtsuThreshold*binWidth,     imageHistogram_magno.rows - cvRound(normHist.at<float>(Magno_OtsuThreshold))),
								cv::Point((Magno_OtsuThreshold+1)*binWidth, 0),
								cv::Scalar::all(0),
								-1, 8, 0 );
				}

				if (bCalc_BMS)
				{
					//@inproceedings{zhang2013saliency,
					// title={Saliency detection: A boolean map approach.},
					// author={Zhang, Jianming and Sclaroff, Stan},
					// booktitle={Proc. of the IEEE International Conference on Computer Vision (ICCV)},
					// year={2013}
					//}

					int SAMPLE_STEP			=	8;//: delta

					/*Note: we transform the kernel width to the equivalent iteration 
					number for OpenCV's **dilate** and **erode** functions**/
					int OPENING_WIDTH		=	2;//: omega_o	
					int DILATION_WIDTH_1	=	3;//: omega_d1
					int DILATION_WIDTH_2	=	11;//: omega_d2

					float BLUR_STD			=	20;//: sigma	
					bool NORMALIZE			=	1;//: whether to use L2-normalization
					bool HANDLE_BORDER		=	0;//: to handle the images with artificial frames

					Mat src_small = retinaOutput_parvo;//inputFrame;
					//resize(inputFrame,src_small,Size(600,inputFrame.rows*(600.0/inputFrame.cols)),0.0,0.0,INTER_AREA);// standard: width: 600 pixel
					//GaussianBlur(src_small,src_small,Size(3,3),1,1);// removing noise

					/* Computing saliency */
					BMS bms(src_small,DILATION_WIDTH_1,OPENING_WIDTH,NORMALIZE,HANDLE_BORDER);
					bms.computeSaliency((float)SAMPLE_STEP);
		
					Mat result = bms.getSaliencyMap();

					/* Post-processing */
					//if (DILATION_WIDTH_2>0)
					//	dilate(result,result,Mat(),Point(-1,-1),DILATION_WIDTH_2);

					if (BLUR_STD > 0)
					{
						int blur_width = MIN(floor(BLUR_STD)*4+1,51);
					//	GaussianBlur(result,result,Size(blur_width,blur_width),BLUR_STD,BLUR_STD);
					}	

					cv::imshow("BMS", result);
				}

				time(&end);
				sec = difftime(end,start);

				fps = frameCount / sec;

				if (1)
				{
					int baseline=0;
					std::ostringstream ossThr, ossCapFps, ossFps;
					ossThr << Magno_OtsuThreshold;
					ossCapFps << videoCapture.get(CAP_PROP_FPS);
					ossFps << fps;
					string text = "Otsu Thr: " + ossThr.str() + "  FPS: " + (ossCapFps > 0 ? (ossCapFps.str() + " (" + ossFps.str() + ")") : ossFps.str());
					cv::Size textSize1 = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
					baseline += thickness;
					cv::Point textOrg1(0, textSize1.height);
					cv::putText(inputFrame, text, textOrg1, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
				}

				cv::imshow	  ("Retina Input",          inputFrame);
				cv::imshow	  ("Retina Parvo",          retinaOutput_parvo);
				cv::imshow	  ("Retina Magno",          retinaOutput_magno);
				cv::imshow	  ("Label frame",			labelFrame);
				cv::imshow	  ("Area Of Interest",      retinaAreaOfInterest);
				//cv::imshow	  ("Magno Otsu Threshold",  retinaThreshold_magno);

				if (bCalc_Histogram)
					cv::imshow	  ("Magno Histogram",		imageHistogram_magno);
 
				cv::moveWindow("Retina Parvo",			((inputFrame.cols+16)*0), ((inputFrame.rows+32)*0));
				cv::moveWindow("Retina Input",			((inputFrame.cols+16)*1), ((inputFrame.rows+32)*0));
				cv::moveWindow("Retina Magno",			((inputFrame.cols+16)*2), ((inputFrame.rows+32)*0));

				cv::moveWindow("Magno Histogram",		((inputFrame.cols+16)*0), ((inputFrame.rows+32)*1));
				cv::moveWindow("Area Of Interest",		((inputFrame.cols+16)*1), ((inputFrame.rows+32)*1));
				cv::moveWindow("Label frame", 			((inputFrame.cols+16)*2), ((inputFrame.rows+32)*1));
				//cv::moveWindow("Magno Otsu Threshold",  ((inputFrame.cols+16)*2), ((inputFrame.rows+32)*1));
            }
 
            if (cv::waitKey(30) >= 0)
                break;
        }
    }
    catch(cv::Exception e)
    {
        const char* err_msg = e.what();
        std::cerr<<"Error using Retina : " << err_msg << std::endl;
        std::cout<<"Error using Retina : " << err_msg << std::endl;
    }
 
    // Program end message
    std::cout << "Retina demo end" << std::endl;
 
    return 0;
}

//
// A Threshold Selection Method from Gray-Level Histograms - NOBUYUKI OTSUA
//
// A nonparametric and unsupervised method of automatic 
// threshold selection for picture segmentation is presented.
// An optimal threshold is selected by the discriminant 
// criterion, namely, so as to maximize the separability 
// of the resultant classes in gray levels.
//
//http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4310076
//
float Px(int init, int end, cv::Mat& retinaHistogram_magno)
{
    float sum = 0.f;
    for (int i = init; i <= end; i++)	sum += retinaHistogram_magno.at<float>(i);
	return (float)sum;
}
float Mx(int init, int end, cv::Mat& retinaHistogram_magno)
{
    float sum = 0.f;
    for (int i = init; i <= end; i++)	sum += i * retinaHistogram_magno.at<float>(i);
	return (float)sum;
}

// Otsu's method
int getOtsuThreshold(cv::Mat& src, float *maxValue)
{
    int t = 0;
         
    cv::Mat p(src);
    if (p.channels() == 3)
        cvtColor(p, p, COLOR_BGR2GRAY);
 
    calcHist(&p, 1, 0, cv::Mat(), retinaHistogram_magno, 1, &histogramSize, 0);
 
    // Loop through all possible t values and maximize between class variance
    vector<float> vec (256);
    float p1,p2,p12;
    float maxVec = 0;
    float maxHist = 0;
 
    if (maxValue)
        *maxValue = 0;
 
    for (int k = 1; k != 255; k++)
    {
        p1 = Px(0, k, retinaHistogram_magno);
        p2 = Px(k + 1, 255, retinaHistogram_magno);
 
        p12 = p1 * p2;
        if (p12 == 0) 
            p12 = 1;
 
        float diff = (Mx(0, k, retinaHistogram_magno) * p2) - (Mx(k + 1, 255, retinaHistogram_magno) * p1);
        vec[k] = (float)diff * diff / p12;
    //  vec[k] = (float)powf((Mx(0, k, retinaHistogram_magno) * p2) - (Mx(k + 1, 255, retinaHistogram_magno) * p1), 2.f) / p12;
 
        if (vec[k] > maxVec)
        {
            maxVec = vec[k];
            t = k; // grab the index
        }
        if (retinaHistogram_magno.at<float>(k) >= maxHist)
        {
            maxHist = retinaHistogram_magno.at<float>(k);
        }
    }
    if (maxValue)
        *maxValue = maxHist;

    return t;

}
 
 
namespace connectedcomponents{
	using std::vector;

	//Find the root of the tree of node i
	template<typename LabelT> inline static
	LabelT findRoot(const vector<LabelT> &P, LabelT i){
		LabelT root = i;
		while(P[root] < root) {
			root = P[root];
		}
		return root;
	}

	//Make all nodes in the path of node i point to root
	template<typename LabelT> inline static
	void setRoot(vector<LabelT> &P, LabelT i, LabelT root){
		while(P[i] < i){
			LabelT j = P[i];
			P[i] = root;
			i = j;
		}
		P[i] = root;
	}

	//Find the root of the tree of the node i and compress the path in the process
	template<typename LabelT> inline static
	LabelT find(vector<LabelT> &P, LabelT i){
		LabelT root = findRoot(P, i);
		setRoot(P, i, root);
		return root;
	}

	//unite the two trees containing nodes i and j and return the new root
	template<typename LabelT> inline static
	LabelT set_union(vector<LabelT> &P, LabelT i, LabelT j){
		LabelT root = findRoot(P, i);
		if(i != j) {
			LabelT rootj = findRoot(P, j);
			if(root > rootj) {
				root = rootj;
			}
			setRoot(P, j, root);
		}
		setRoot(P, i, root);
		return root;
	}

	//Flatten the Union Find tree and relabel the components
	template<typename LabelT> inline static
	LabelT flattenL(vector<LabelT> &P){
		LabelT k = 1;
		for(size_t i = 1; i < P.size(); ++i){
			if(P[i] < i){
				P[i] = P[P[i]];
			}
			else{
				P[i] = k; 
				k = k + 1;
			}
		}
		return k;
	}

	const int G4[2][2] = {{-1, 0}, {0, -1}};//b, d neighborhoods
	const int G8[4][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};//a, b, c, d neighborhoods

	//Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) 
	// variant using decision trees, Kesheng Wu, et al
	template<typename LabelT, typename PixelT, int connectivity = 8>
	struct LabelingImpl{
		LabelT operator()(cv::Mat &L, const cv::Mat &I){
			const int rows = L.rows;
			const int cols = L.cols;
			size_t nPixels = size_t(rows) * cols;
			vector<LabelT> P; 
			P.push_back(0);
			LabelT l = 1;

			//scanning phase
			for(int r_i = 0; r_i < rows; ++r_i){
				for(int c_i = 0; c_i < cols; ++c_i){
					if(!I.at<PixelT>(r_i, c_i)){
						L.at<LabelT>(r_i, c_i) = 0;
						continue;
					}
					if(connectivity == 8){
						const int a = 0;
						const int b = 1;
						const int c = 2;
						const int d = 3;
						bool T[4];

						for(size_t i = 0; i < 4; ++i){
							int gr = r_i + G8[i][0];
							int gc = c_i + G8[i][1];
							T[i] = false;

							if(gr >= 0 && gr < I.rows && gc >= 0 && gc < I.cols){
								if(I.at<PixelT>(gr, gc)){
									T[i] = true;
								}
							}
						}

						//decision tree
						if(T[b]){
							//copy(b)
							L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[b][0], c_i + G8[b][1]);
						}else{//not b
							if(T[c]){
								if(T[a]){
									//copy(c, a)
									L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]), L.at<LabelT>(r_i + G8[a][0], c_i + G8[a][1]));
								}else{
									if(T[d]){
										//copy(c, d)
										L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]), L.at<LabelT>(r_i + G8[d][0], c_i + G8[d][1]));
									}else{
										//copy(c)
										L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[c][0], c_i + G8[c][1]);
									}
								}
							}else{//not c
								if(T[a]){
									//copy(a)
									L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[a][0], c_i + G8[a][1]);
								}else{
									if(T[d]){
										//copy(d)
										L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G8[d][0], c_i + G8[d][1]);
									}else{
										//new label
										L.at<LabelT>(r_i, c_i) = l;
										P.push_back(l);//P[l] = l;
										l = l + 1;
									}
								}
							}
						}
					}else{
						//B & D only
						const int b = 0;
						const int d = 1;
						CV_Assert(connectivity == 4);
						bool T[2];
						for(size_t i = 0; i < 2; ++i){
							int gr = r_i + G4[i][0];
							int gc = c_i + G4[i][1];
							T[i] = false;
							if(gr >= 0 && gr < I.rows && gc >= 0 && gc < I.cols){
								if(I.at<PixelT>(gr, gc)){
									T[i] = true;
								}
							}
						}
						if(T[b]){
							if(T[d]){
								//copy(d, b)
								L.at<LabelT>(r_i, c_i) = set_union(P, L.at<LabelT>(r_i + G4[d][0], c_i + G4[d][1]), L.at<LabelT>(r_i + G4[b][0], c_i + G4[b][1]));
							}else{
								//copy(b)
								L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G4[b][0], c_i + G4[b][1]);
							}
						}else{
							if(T[d]){
								//copy(d)
								L.at<LabelT>(r_i, c_i) = L.at<LabelT>(r_i + G4[d][0], c_i + G4[d][1]);
							}else{
								//new label
								L.at<LabelT>(r_i, c_i) = l;
								P.push_back(l);//P[l] = l;
								l = l + 1;
							}
						}
					}
				}
			}


			//analysis
			LabelT nLabels = flattenL(P);

			//assign final labels
			for(int r = 0; r < L.rows; ++r){
				for(int c = 0; c < L.cols; ++c){
					L.at<LabelT>(r, c) = P[L.at<LabelT>(r, c)];
				}
			}
			return nLabels;
		}//End function LabelingImpl operator()
	};//End struct LabelingImpl
}//end namespace connectedcomponents

//L's type must have an appropriate depth for the number of pixels in I
int connectedComponents(cv::Mat &L, const cv::Mat &I, int connectivity)
{
	//int lc = L.channels();
	//int ic = I.channels();
    
	CV_Assert(L.rows == I.rows);
	CV_Assert(L.cols == I.cols);
	CV_Assert(L.channels() == 1);
	CV_Assert(I.channels() == 1);
	CV_Assert(connectivity == 8 || connectivity == 4);

	int lDepth = L.depth();
	int iDepth = I.depth();
	using connectedcomponents::LabelingImpl;

	//warn if L's depth is not sufficient?
	if(lDepth == CV_8U){
		if(iDepth == CV_8U || iDepth == CV_8S){
			if(connectivity == 4){
				return LabelingImpl<uint8_t, uint8_t, 4>()(L, I);
			}else{
				return LabelingImpl<uint8_t, uint8_t, 8>()(L, I);
			}
		}else if(iDepth == CV_16U || iDepth == CV_16S){
			if(connectivity == 4){
				return LabelingImpl<uint8_t, uint16_t, 4>()(L, I);
			}else{
				return LabelingImpl<uint8_t, uint16_t, 8>()(L, I);
			}
		}else if(iDepth == CV_32S){
			if(connectivity == 4){
				return LabelingImpl<uint8_t, int32_t, 4>()(L, I);
			}else{
				return LabelingImpl<uint8_t, int32_t, 8>()(L, I);
			}
		}else if(iDepth == CV_32F){
			if(connectivity == 4){
				return LabelingImpl<uint8_t, float, 4>()(L, I);
			}else{
				return LabelingImpl<uint8_t, float, 8>()(L, I);
			}
		}else if(iDepth == CV_64F){
			if(connectivity == 4){
				return LabelingImpl<uint8_t, double, 4>()(L, I);
			}else{
				return LabelingImpl<uint8_t, double, 8>()(L, I);
			}
		}
		}else if(lDepth == CV_16U){
			if(iDepth == CV_8U || iDepth == CV_8S){
				if(connectivity == 4){
					return LabelingImpl<uint16_t, uint8_t, 4>()(L, I);
				}else{
					return LabelingImpl<uint16_t, uint8_t, 8>()(L, I);
				}
			}else if(iDepth == CV_16U || iDepth == CV_16S){
				if(connectivity == 4){
					return LabelingImpl<uint16_t, uint16_t, 4>()(L, I);
				}else{
					return LabelingImpl<uint16_t, uint16_t, 8>()(L, I);
				}
			}else if(iDepth == CV_32S){
				if(connectivity == 4){
					return LabelingImpl<uint16_t, int32_t, 4>()(L, I);
				}else{
					return LabelingImpl<uint16_t, int32_t, 8>()(L, I);
				}
			}else if(iDepth == CV_32F){
				if(connectivity == 4){
					return LabelingImpl<uint16_t, float, 4>()(L, I);
				}else{
					return LabelingImpl<uint16_t, float, 8>()(L, I);
				}
			}else if(iDepth == CV_64F){
				if(connectivity == 4){
					return LabelingImpl<uint16_t, double, 4>()(L, I);
				}else{
					return LabelingImpl<uint16_t, double, 8>()(L, I);
				}
			}
		}else if(lDepth == CV_32S){
			if(iDepth == CV_8U || iDepth == CV_8S){
				if(connectivity == 4){
					return LabelingImpl<int32_t, uint8_t, 4>()(L, I);
				}else{
					return LabelingImpl<int32_t, uint8_t, 8>()(L, I);
				}
			}else if(iDepth == CV_16U || iDepth == CV_16S){
				if(connectivity == 4){
					return LabelingImpl<int32_t, uint16_t, 4>()(L, I);
				}else{
					return LabelingImpl<int32_t, uint16_t, 8>()(L, I);
			}
			}else if(iDepth == CV_32S){
				if(connectivity == 4){
					return LabelingImpl<int32_t, int32_t, 4>()(L, I);
				}else{
					return LabelingImpl<int32_t, int32_t, 8>()(L, I);
				}
			}else if(iDepth == CV_32F){
				if(connectivity == 4){
					return LabelingImpl<int32_t, float, 4>()(L, I);
				}else{
					return LabelingImpl<int32_t, float, 8>()(L, I);
				}
			}else if(iDepth == CV_64F){
				if(connectivity == 4){
					return LabelingImpl<int32_t, double, 4>()(L, I);
				}else{
					return LabelingImpl<int32_t, double, 8>()(L, I);
			}
		}
	}

	CV_Error(Error::Code::StsUnsupportedFormat, "unsupported label/image type");
	return -1;
}
