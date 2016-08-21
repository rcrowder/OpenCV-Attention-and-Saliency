# Saliency and Attention focussing tests using OpenCV

## Background

Autumn 2014  
Before OpenCV Saliency API  
During dicovery of ML and MI techniques  
 
## Requirements

Git, CMake, C++ compiler, OpenCV 3.x with contrib modules


## References

		// http://docs.opencv.org/doc/tutorials/contrib/retina_model/retina_model.html

					//@inproceedings{zhang2013saliency,
					// title={Saliency detection: A boolean map approach.},
					// author={Zhang, Jianming and Sclaroff, Stan},
					// booktitle={Proc. of the IEEE International Conference on Computer Vision (ICCV)},
					// year={2013}
					//}
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

	//Based on "Two Strategies to Speed up Connected Components Algorithms", 
	// the SAUF (Scan array union find) variant using decision trees, 
	// Kesheng Wu, et al

*	Implemetation of the saliency detction method described in paper
*	"Saliency Detection: A Boolean Map Approach", Jianming Zhang, 
*	Stan Sclaroff, ICCV, 2013

A. Erdem and E. Erdem. (2003): Visual saliency estimation by nonlinearly integrating features using region covariances
http://www.journalofvision.org/content/13/4/11.full.pdf

B. Celikkale, A. Erdem and E. Erdem (2013): Visual Attention-driven Spatial Pooling for Image Memorability.
IEEE Computer Vision and Pattern Recognition Workshops (CVPRW), Portland, Oregon, USA, June 2013.
http://web.cs.hacettepe.edu.tr/~aykut/projects/saliencymemorability/index.html  

Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(11), 1254–1259.

@misc{mit-saliency-benchmark,
  author       = {Zoya Bylinskii and Tilke Judd and Fr{\'e}do Durand and Aude Oliva and Antonio Torralba},
  title        = {MIT Saliency Benchmark},
  howpublished = {http://saliency.mit.edu/}
}

B. Celikkale, A. Erdem and E. Erdem (2013): Visual Attention-driven Spatial Pooling for Image Memorability  
IEEE Computer Vision and Pattern Recognition Workshops (CVPRW), Portland, Oregon, USA, June 2013.  

Cui Y, Liu DL, McFarland J, Pack CC, Butts DA (2013) Modulation of stimulus processing by network activity across cortical depth  
MT. Soc. Neurosci. Abs. 39:359, San Diego, USA.  
https://dl.dropboxusercontent.com/u/51202818/website/SfN2013_MTposter.pdf  

http://terpconnect.umd.edu/~ywcui/Yuweis_Homepage/Publication.html

		// http://ps.is.tue.mpg.de/project/MPI_Sintel_Flow
		// MPI Sintel Flow 'testing' data set

		//@inproceedings{Wulff:ECCVws:2012,
		// title = {Lessons and insights from creating a synthetic optical flow benchmark},
		// author = {Wulff, J. and Butler, D. J. and Stanley, G. B. and Black, M. J.},
		// booktitle = {ECCV Workshop on Unsolved Problems in Optical Flow and Stereo Estimation},
		// editor = {{A. Fusiello et al. (Eds.)}},
		// publisher = {Springer-Verlag},
		// series = {Part II, LNCS 7584},
		// month = oct,
		// pages = {168--177},
		// year = {2012}
		//}

		// http://vision.middlebury.edu/flow/data/
