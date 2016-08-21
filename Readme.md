# Saliency and Attention tests using OpenCV

## Background

Autumn 2014  
Before OpenCV Saliency API  
During discovery of ML and MI techniques  

http://www.scholarpedia.org/article/Saliency_map

## Description

OpenCV Retina module [1], particularly Alexandre Benoit et al. paper [2] - 
> The proposed model originates from Jeanny Herault’s research at Gipsa. It is involved in image processing applications with Listic (code maintainer) lab. This is not a complete model but it already present interesting properties that can be involved for enhanced image processing experience. The model allows the following human retina properties to be used :
> * spectral whitening that has 3 important effects: high spatio-temporal frequency signals canceling (noise), mid-frequencies details enhancement and low frequencies luminance energy reduction. This all in one property directly allows visual signals cleaning of classical undesired distortions introduced by image sensors and input luminance range.
> * local logarithmic luminance compression allows details to be enhanced even in low light conditions.
> * decorrelation of the details information (Parvocellular output channel) and transient information (events, motion made available at the Magnocellular output channel).

## Requirements

Git, [CMake](https://cmake.org/), C++ compiler/toolchain, [OpenCV 3.x](https://github.com/opencv/opencv) built and configured with the [contrib modules](https://github.com/opencv/opencv_contrib).

## Datasets

* CVonline: The Evolving, Distributed, Non-Proprietary, On-Line Compendium of Computer Vision - <http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm>

* MIT Saliency Benchmark - <http://saliency.mit.edu/>

* Middlebury Optical Flow dataset - <http://vision.middlebury.edu/flow/>

* Sintel Optical Flow dataset - <http://ps.is.tue.mpg.de/research_projects/mpi-sintel-flow>

## References

1. OpenCV Retina module - <http://docs.opencv.org/doc/tutorials/contrib/retina_model/retina_model.html>

2. Benoit A., Caplier A., Durette B., Herault, J., “Using Human Visual System Modeling For Bio-Inspired Low Level Image Processing”, Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773. DOI <http://dx.doi.org/10.1016/j.cviu.2010.01.011>

3. "Saliency Detection: A Boolean Map Approach", Jianming Zhang, Stan Sclaroff, ICCV, 2013

4. "A Threshold Selection Method from Gray-Level Histograms", Nobuyuki Otsua.  
A nonparametric and unsupervised method of automatic threshold selection for picture segmentation is presented. An optimal threshold is selected by the discriminant  criterion, namely, so as to maximize the separability of the resultant classes in gray levels.  
<http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4310076>

5. "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant using decision trees, Kesheng Wu, et al

6. Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(11), 1254–1259.

7. A. Erdem and E. Erdem. (2003): Visual saliency estimation by nonlinearly integrating features using region covariances
<http://www.journalofvision.org/content/13/4/11.full.pdf>

8. B. Celikkale, A. Erdem and E. Erdem (2013): Visual Attention-driven Spatial Pooling for Image Memorability.
IEEE Computer Vision and Pattern Recognition Workshops (CVPRW), Portland, Oregon, USA, June 2013.
<http://web.cs.hacettepe.edu.tr/~aykut/projects/saliencymemorability/index.html>  

9. Cui Y, Liu DL, McFarland J, Pack CC, Butts DA (2013) Modulation of stimulus processing by network activity across cortical depth  
MT. Soc. Neurosci. Abs. 39:359, San Diego, USA.  
<https://dl.dropboxusercontent.com/u/51202818/website/SfN2013_MTposter.pdf>  

<http://terpconnect.umd.edu/~ywcui/Yuweis_Homepage/Publication.html>
