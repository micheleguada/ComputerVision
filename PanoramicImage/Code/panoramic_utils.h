// Computer Vision 2021 (P. Zanuttigh) - LAB 4 
//
// Extended while completing the Final project by: 
//      Michele Guadagnini - Mt. 1230663

#ifndef LAB4__PANORAMIC__UTILS__H
#define LAB4__PANORAMIC__UTILS__H

#include <memory>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>


class PanoramicUtils
{
public:
    
    // project an image on the cylindrical surface in gray_scale
    static cv::Mat cylindricalProj( const cv::Mat& image, 
                                    const double angle 
                                  );
    
// Added functions with respect to the provided library -----------------------------------------------------
    
    // project an image on the cylinder in RGB 
    static cv::Mat cylindricalProjRGB( const cv::Mat& image, 
                                       const double angle 
                                     );
    
    // histogram equalization method
    static cv::Mat intensity_hist_EQ( const cv::Mat& input );

    // extract SIFT descriptors from an image
    static void SIFTkeypoints( cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor );
    
    // find matching features between different images
    static std::vector<cv::DMatch> SIFTmatches( cv::Mat &desc1, cv::Mat &desc2, double ratio );
    
    // use RANSAC to estimate translations between adiacent images
    static cv::Mat find_translation( std::vector<cv::KeyPoint>& kps1, 
                                     std::vector<cv::KeyPoint>& kps2,
                                     std::vector<cv::DMatch>& matches
                                   );   
    
    // build and visualize / store the panoramic image
    static cv::Mat build_panoramic( std::vector<cv::Mat> &images,
                                    std::vector<cv::Mat> &transforms
                                  );  
};

#endif // LAB4__PANORAMIC__UTILS__H
