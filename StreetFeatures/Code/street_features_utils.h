// Computer Vision 2021 - LAB 3 
//
// Library containing functions to detect street lanes and round signs; 
//
// Author: Michele Guadagnini - Mt. 1230663 

#ifndef STREET__FEATURES__H
#define STREET__FEATURES__H

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


class StreetFeaturesUtils
{
public:
    
    // Edges detection with Canny
    static cv::Mat DetectEdges( cv::Mat & image );  
    
    // Lines retrieving with Hough transform
    static std::vector<cv::Vec3f> RetrieveLines( cv::Mat& image,
                                                 cv::Mat& edges
                                               );
    
    // Round signs detection with Hough transform
    static std::vector<cv::Vec4f> RetrieveCircles( cv::Mat& image,
                                                   double sigma
                                                 );
    
    // Plot and save the image with detected features overlapped
    static void SaveResult( cv::Mat& input_img, 
                            std::vector<cv::Vec3f> lines, 
                            std::vector<cv::Vec4f> circles, 
                            std::string outname,
                            std::string plot_option
                          );
    
    // Draw lines above image
    static cv::Mat DrawLines( std::vector<cv::Vec3f>& lines, 
                              cv::Mat& img 
                            );
    
    // Filter the image before appling detectors
    static cv::Mat ApplyFilter( cv::Mat& image, 
                                double sigma
                              );
    
    // Compute polygon between detected lines
    static std::vector<cv::Point> ComputePolyVertices( std::vector<cv::Vec3f>& lines,
                                                       cv::Size img_size
                                                     );
    
};


// class used to pass parameters and objects to callback functions
class TrackbarParams
{
public:
    
    cv::Mat image, filtered, edges;
    std::vector<cv::Vec3f> lines;
    std::vector<cv::Vec4f> circles;
    
    std::string window;
    std::vector<std::string> tb_names;
    
};

#endif  // STREET__FEATURES__H
