// Computer Vision 2021 (P. Zanuttigh) - LAB 4 
//
// Extended while completing the Final project by: 
//      Michele Guadagnini - Mt. 1230663

#include "panoramic_utils.h"
#include <memory>
#include <iostream>
#include <algorithm> 
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>


    // Projection of an image on the cylinder in gray_scale 
    cv::Mat PanoramicUtils::cylindricalProj( const cv::Mat& image, 
                                             const double angle 
                                           )
    {
        cv::Mat tmp, result;
        if (image.channels() == 3) {
            cv::cvtColor(image, tmp, cv::COLOR_BGR2GRAY);
        }        
        result = tmp.clone();

        double alpha(angle / 180 * CV_PI);
        double d((image.cols / 2.0) / tan(alpha));
        double r(d / cos(alpha));
        double d_by_r(d / r);
        int half_height_image(image.rows / 2);
        int half_width_image(image.cols / 2);

        for (int x = -half_width_image + 1,
             x_end = half_width_image; x < x_end; ++x)
        {
            for (int y = -half_height_image + 1,
                 y_end = half_height_image; y < y_end; ++y)
            {
                double x1(d * tan(x / r));
                double y1(y * d_by_r / cos(x / r));

                if (x1 < half_width_image &&
                    x1 > -half_width_image + 1 &&
                    y1 < half_height_image &&
                    y1 > -half_height_image + 1)
                {
                    result.at<uchar>(y + half_height_image, x + half_width_image)
                        = tmp.at<uchar>(round(y1 + half_height_image),
                            round(x1 + half_width_image));
                }
            }
        }

        return result;
    }  
    
// Added functions with respect to the provided library ---------------------------------------------------------------
    
    // Projection of an image on the cylinder in RGB (3 channels)
    cv::Mat PanoramicUtils::cylindricalProjRGB( const cv::Mat& image, 
                                                const double angle 
                                              )
    {
        if (image.channels() != 3) {
            std::cout << "cylindricalProjRGB: the image does not have 3 channels. Skipping projection..." << std::endl;
            return image.clone();
        }
        
        std::vector<cv::Mat> channels(3);
        std::vector<cv::Mat> result(3);
        
        // splitting channels
        cv::split(image, channels);
        for (int ll=0; ll < 3; ll++) {
            result.at(ll) = channels.at(ll).clone();
        }
        
        double alpha(angle / 180 * CV_PI);
        double d((image.cols / 2.0) / tan(alpha));
        double r(d / cos(alpha));
        double d_by_r(d / r);
        int half_height_image(image.rows / 2);
        int half_width_image(image.cols / 2);
        
        for (int x = -half_width_image + 1,
            x_end = half_width_image; x < x_end; ++x)
        {
            for (int y = -half_height_image + 1,
                y_end = half_height_image; y < y_end; ++y)
            {
                double x1(d * tan(x / r));
                double y1(y * d_by_r / cos(x / r));

                if (x1 < half_width_image &&
                    x1 > -half_width_image + 1 &&
                    y1 < half_height_image &&
                    y1 > -half_height_image + 1)
                {
                    // iterate over channels
                    for (int kk=0; kk < 3; kk++) {
                        result.at(kk).at<uchar>(y + half_height_image, x + half_width_image)
                            = channels.at(kk).at<uchar>(round(y1 + half_height_image),
                                round(x1 + half_width_image));
                    }
                }
            }
        }
        
        // merging again the 3 channels
        cv::Mat RGBproj;
        cv::merge(result, RGBproj);
        
        return RGBproj;
    }


    // Histogram equalization method
    cv::Mat PanoramicUtils::intensity_hist_EQ( const cv::Mat& input )
    {
        if(input.channels() == 3)       //RGB
        {
            cv::Mat ycrcb;
            cv::cvtColor(input, ycrcb, cv::COLOR_BGR2YCrCb);

            std::vector<cv::Mat> channels;
            cv::split(ycrcb, channels);

            // equalization is done on the intensity channel
            cv::equalizeHist(channels[0], channels[0]);

            cv::Mat result;
            cv::merge(channels, ycrcb);
            cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);

            return result;
        }
        else {                          //gray-scale
            cv::Mat result;
            cv::equalizeHist(input, result);
            
            return result;
        }
    }
    
    
    // Extract and compute SIFT descriptors from an image
    void PanoramicUtils::SIFTkeypoints( cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptor)
    {
        cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
        siftPtr->detectAndCompute(img, cv::noArray(), keypoints, descriptor);

        return;
    }
    
    
    // Find matching features between different images
    std::vector<cv::DMatch> PanoramicUtils::SIFTmatches( cv::Mat &desc1, cv::Mat &desc2, double ratio )
    {
        std::vector<cv::DMatch> matches;
        
        // computing the matches
        cv::Ptr<cv::BFMatcher> bfmPtr = cv::BFMatcher::create(cv::NORM_L2);
        bfmPtr->match( desc1, desc2, matches );
        
        // computing minimum distance
        double min_dist = 1000000;
        for (int kk=0; kk < matches.size(); kk++) {
            if (matches.at(kk).distance < min_dist) {
                 min_dist = matches.at(kk).distance;
            }
        }            
        
        std::vector<cv::DMatch> filtered_matches(matches.size());
        // selecting best matches
         auto it = std::copy_if( matches.begin(), 
                                 matches.end(), 
                                 filtered_matches.begin(), 
                                 [&]( cv::DMatch mm ){ return (mm.distance < ratio*min_dist); } 
                               );
         
        // shrinking the vector to the new size after filtering
        filtered_matches.resize(std::distance(filtered_matches.begin(), it));    
        
        return filtered_matches;
    }
    
    
    // Use RANSAC to estimate translations/transformation between adiacent images
    cv::Mat PanoramicUtils::find_translation( std::vector<cv::KeyPoint>& kps1, 
                                              std::vector<cv::KeyPoint>& kps2,
                                              std::vector<cv::DMatch>& matches
                                            )   
    {
        cv::Mat hom;
        std::vector<cv::Point2d> pts1, pts2;

        // use matches to select good points
        for ( int ii=0; ii < matches.size(); ii++ ) {
            pts1.push_back( kps1[ matches[ii].queryIdx ].pt );
            pts2.push_back( kps2[ matches[ii].trainIdx ].pt );
        }
        
        // use RANSAC to find inliers / homography
        std::vector<uchar> mask_inliers;
        hom = cv::findHomography( pts1, pts2, mask_inliers, cv::RANSAC );
        
        // extract the inliers
        std::vector<cv::Point2d> inliers1, inliers2;
        for (int ii=0; ii < mask_inliers.size(); ii++) {
            if (mask_inliers.at(ii) == 1) {
                inliers1.push_back(pts1.at(ii));
                inliers2.push_back(pts2.at(ii));      
            }
        }

        // compute the average dx and dy for translation
        double sum_x = 0;
        double sum_y = 0;
        for (int jj=0; jj < inliers1.size(); jj++) {
            sum_x += (inliers1.at(jj).x - inliers2.at(jj).x);
            sum_y += (inliers1.at(jj).y - inliers2.at(jj).y);
        }
        
        double avg_x( sum_x/inliers1.size() );
        double avg_y( sum_y/inliers1.size() );

        // build translation matrix
        double tmp2D[6] = {1., 0., avg_x, 0., 1., avg_y};
        cv::Mat trsl = cv::Mat(2, 3, CV_64FC1, &tmp2D); 
        
        return trsl.clone();
    }
    
    // Build and return the panoramic image
    cv::Mat PanoramicUtils::build_panoramic( std::vector<cv::Mat> &images,
                                             std::vector<cv::Mat> &transforms
                                           )
    {
        cv::Mat panoramic = images.at(0).clone();
        cv::Mat trl_result;
        
        // initialize the first transform (identity)
        double tmp2D[6] = {1., 0., 0., 0., 1., 0.};
        cv::Mat combined_trsl = cv::Mat(2, 3, CV_64FC1, &tmp2D).clone(); 
        
        // matrix row to change transformations into homogeneous coordinates
        double aux[3] = {0., 0., 1.};
        cv::Mat bottom_row = cv::Mat(1, 3, CV_64FC1, &aux).clone();
        
        int size_growth = 0;
        double max_y_trl = 0.;
        double min_y_trl = 1.;
        
        for (int ii=0; ii < images.size()-1; ii++) {
                      
            // get new translation length on x axis            
            double trl_x = transforms.at(ii).at<double>(0,2);
            int t_x = static_cast<int>(trl_x);
            
            // compute new image size
            size_growth += t_x;
            
            // combine translations using homogeneous coordinates
            cv::Mat comb_hmg   = combined_trsl;
            comb_hmg.push_back(bottom_row);
            cv::Mat transf_hmg = transforms.at(ii);
            transf_hmg.push_back(bottom_row);
            
            combined_trsl = comb_hmg * transf_hmg;            
            combined_trsl = combined_trsl( cv::Rect(0, 0, 3, 2) );
            
            // apply combined translation
            cv::warpAffine( images.at(ii+1),   trl_result,   combined_trsl, 
                            cv::Size(images.at(0).cols + size_growth, images.at(0).rows), 
                            cv::INTER_LINEAR);
            
            // add previous result
            cv::Mat half( trl_result, cv::Rect(0, 0, images.at(0).cols + size_growth - t_x, images.at(0).rows) );
            panoramic.copyTo(half);
            
            // store new panoramic with added one image
            panoramic = trl_result.clone();
            
            // get max and min y translations for cropping
            max_y_trl = std::max(combined_trsl.at<double>(1,2), max_y_trl);
            min_y_trl = std::min(combined_trsl.at<double>(1,2), min_y_trl);
            
        } 
        
        // cut on y-axis to hide vertical translations effects
        int crop_top    = static_cast<int>(max_y_trl+1.5);
        int crop_bottom = std::abs( std::min( static_cast<int>(min_y_trl-1.5), 0 ) );
        
        panoramic = panoramic( cv::Rect( 0,              
                                         crop_top, 
                                         panoramic.cols, 
                                         panoramic.rows - (crop_bottom+crop_top) )
                             );
        
        return panoramic;
    }
    
