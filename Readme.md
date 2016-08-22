# Saliency and Attention tests using OpenCV

## Background

In the Autumn of 2014 I had a simple robotics platform that included a [CMUCam3](http://www.cmucam.org/projects/cmucam3) single sensor camera module mounted on a pan-tilt servo frame. With that inplace I set about discovering ways to analyse the video stream from the camera. This led to the discovery of the contributed Retina module [1] in the Open Computer Vision (OpenCV) library. Note; this was before the addition of the OpenCV Saliency API.

It appealing due to the biologically inspired splitting of motion and detail visual information, i.e. magno and parvo cellular pathways into cortical area (V1), via the lateral geniculate nucleus (LGN) of the thalamus [1].

* Parvocellular ganglion retinal cells being smaller, slower, and carry many details such as colour.
* Magnocellular ganglion retinal cells being larger, faster, rather rough in their representations, and carrying transient information such as motion details.

The idea was to process the magnocellular output to create [saliency maps](http://www.scholarpedia.org/article/Saliency_map) as a way of detecting areas of interest to pay attention to.

Work here was interrupted/side-tracked due to a furthering interest into sub-cortical auditory processing on the path from ears into cortical area A1. As well as further research work into fast optical flow techniques.

## Description

The OpenCV Retina module [2] is described as such; further information can be deduced from Alexandre Benoit et al. paper [3] - 
> The proposed model originates from Jeanny Herault’s research at Gipsa. It is involved in image processing applications with Listic (code maintainer) lab. This is not a complete model but it already present interesting properties that can be involved for enhanced image processing experience. The model allows the following human retina properties to be used :
> * spectral whitening that has 3 important effects: high spatio-temporal frequency signals canceling (noise), mid-frequencies details enhancement and low frequencies luminance energy reduction. This all in one property directly allows visual signals cleaning of classical undesired distortions introduced by image sensors and input luminance range.
> * local logarithmic luminance compression allows details to be enhanced even in low light conditions.
> * decorrelation of the details information (Parvocellular output channel) and transient information (events, motion made available at the Magnocellular output channel).

The Retina module has a simple interface that hides away the retinal processing, before final output of parvo and magno cellular images. It relies upon video streams for it's cellular network to produce appropriate results.

Subsequent processing of the grey-scale magno cellular image uses:
* [Otsu's method](https://en.wikipedia.org/wiki/Otsu%27s_method) "A nonparametric and unsupervised method of automatic threshold selection for picture segmentation is presented. An optimal threshold is selected by the discriminant criterion, namely, so as to maximize the separability of the resultant classes in gray levels." [4].
* Zhang and Sclaroff's Boolean Map based Saliency model (BMS) [5]. That compares favourably against other approaches, such as Itti et al. [6] or Erdem et al. [7].
* With Kesheng Wu et al. work on speeding up connected component labeling [8].

Future work is required to further determination of appropriate motion features, and inclusion of [computational models of visual attention](http://www.scholarpedia.org/article/Computational_models_of_visual_attention).

## Requirements

Git, [CMake](https://cmake.org/), C++ compiler/toolchain, [OpenCV 3.x](https://github.com/opencv/opencv) built and configured with the [contrib modules](https://github.com/opencv/opencv_contrib).

## Datasets

* CVonline: The Evolving, Distributed, Non-Proprietary, On-Line Compendium of Computer Vision - <http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm>

* MIT Saliency Benchmark - <http://saliency.mit.edu/>

* Middlebury Optical Flow dataset - <http://vision.middlebury.edu/flow/>

* Sintel Optical Flow dataset - <http://ps.is.tue.mpg.de/research_projects/mpi-sintel-flow>

## References

1. Maunsell, J.H. (1992). Functional visual streams. Current Opinion in Neurobiology, 2, 506–510.

2. OpenCV Retina module - <http://docs.opencv.org/3.1.0/d3/d86/tutorial_bioinspired_retina_model.html>

3. Benoit A., Caplier A., Durette B., Herault, J., “Using Human Visual System Modeling For Bio-Inspired Low Level Image Processing”, Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773. DOI <http://dx.doi.org/10.1016/j.cviu.2010.01.011> <https://sites.google.com/site/benoitalexandrevision/>

4. "A Threshold Selection Method from Gray-Level Histograms", Nobuyuki Otsua (1979). <http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4310076>

5. "Saliency Detection: A Boolean Map Approach", Jianming Zhang, Stan Sclaroff, ICCV, 2013 <http://cs-people.bu.edu/jmzhang/BMS/BMS.html>

6. Itti, L., Koch, C., & Niebur, E. (1998). A model of saliency-based visual attention for rapid scene analysis. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(11), 1254–1259.

7. A. Erdem and E. Erdem. (2003): Visual saliency estimation by nonlinearly integrating features using region covariances
<http://www.journalofvision.org/content/13/4/11.full.pdf>

8. "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant using decision trees, Kesheng Wu, et al. <http://crd-legacy.lbl.gov/~kewu/ps/LBNL-59102.html>

9. B. Celikkale, A. Erdem and E. Erdem (2013): Visual Attention-driven Spatial Pooling for Image Memorability.
IEEE Computer Vision and Pattern Recognition Workshops (CVPRW), Portland, Oregon, USA, June 2013.
<http://web.cs.hacettepe.edu.tr/~aykut/projects/saliencymemorability/index.html>  

10. Bruce, N.D.B., Tsotsos, J.K. (2009). Saliency, Attention, and Visual Search: An Information Theoretic Approach, Journal of Vision 9:3, p1-24.

11. Cui Y, Liu DL, McFarland J, Pack CC, Butts DA (2013) Modulation of stimulus processing by network activity across cortical depth  
MT. Soc. Neurosci. Abs. 39:359, San Diego, USA.  
<https://dl.dropboxusercontent.com/u/51202818/website/SfN2013_MTposter.pdf> <http://terpconnect.umd.edu/~ywcui/Yuweis_Homepage/Publication.html>
