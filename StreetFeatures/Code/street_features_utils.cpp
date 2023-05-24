// Computer Vision 2021 - LAB 3 
//
// Library containing functions to detect street lanes and round signs; 
//
// Author: Michele Guadagnini - Mt. 1230663 

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "street_features_utils.h"

// define colors for features
const cv::Scalar line_color   = cv::Scalar(0, 0, 255),
                 circle_color = cv::Scalar(0, 255, 0),
                 region_color = cv::Scalar(255, 0, 255);

                 
                 
// ###################### CALLBACK FUNCTIONS ########################

// callback function for Canny thresholds trackbars
static void onCannyTrackbar( int x, void* data ) {   
    
    // recasting userdata
    TrackbarParams *pars = (TrackbarParams*)data;
    
    // getting Canny thresholds from trackbars
    int low  = cv::getTrackbarPos( pars->tb_names[0], pars->window );   
    int high = cv::getTrackbarPos( pars->tb_names[1], pars->window );
    
    // apply Canny
    cv::Canny( pars->filtered, pars->edges, low, high );
    
    // plot the edges
    cv::imshow( pars->window, pars->edges );
    
    return;
}


// callback function for Hough transform trackbars for lines detection
static void onLinesTrackbar( int x, void* data ) {
    
    // recasting userdata
    TrackbarParams* pars = (TrackbarParams*)data;
    
    // getting Canny thresholds from trackbars
    int rho_res   = cv::getTrackbarPos( pars->tb_names[0], pars->window );   
    int theta_res = cv::getTrackbarPos( pars->tb_names[1], pars->window );
    int count_th  = cv::getTrackbarPos( pars->tb_names[2], pars->window );
    int min_theta = cv::getTrackbarPos( pars->tb_names[3], pars->window );
    int max_theta = cv::getTrackbarPos( pars->tb_names[4], pars->window );
    
    // ensure min-max order
    if (min_theta >= max_theta) {
        min_theta = max_theta -1;
        std::cout << "  Error: Min. theta must be smaller than Max. theta\n";
        std::cout << "     -> Min. theta set to: Max. theta -1\n"; 
    }
    
    // apply hough transform
    cv::HoughLines( pars->edges, pars->lines, rho_res, theta_res*(CV_PI/180), count_th, 
                    0, 0, min_theta*(CV_PI/180), max_theta*(CV_PI/180) );
    
    // draw the lines 
    cv::Mat drawed = StreetFeaturesUtils::DrawLines( pars->lines, pars->image );
    cv::imshow( pars->window, drawed );
    
    return;
}


// callback function for Hough transform trackbars for circles detection
static void onCirclesTrackbar( int x, void* data ) {
    
    // recasting userdata
    TrackbarParams* pars = (TrackbarParams*)data;
    
    // getting Canny thresholds from trackbars
    int minDist = cv::getTrackbarPos( pars->tb_names[0], pars->window );
    int th_high = cv::getTrackbarPos( pars->tb_names[1], pars->window );
    int th_acc  = cv::getTrackbarPos( pars->tb_names[2], pars->window );
    int minR    = cv::getTrackbarPos( pars->tb_names[3], pars->window );  
    int maxR    = cv::getTrackbarPos( pars->tb_names[4], pars->window );   
    
    // reset circles
    pars->circles.clear();
    
    // apply hough transform
    cv::HoughCircles( pars->filtered, pars->circles, cv::HOUGH_GRADIENT, 1, minDist, th_high, th_acc, minR, maxR);
    
    // draw the circles
    cv::Mat drawed = pars->image.clone();
    int thickness = 2;
    
    for( int j=0; j<pars->circles.size(); j++) {
        cv::Point center = cv::Point( cvRound(pars->circles[j][0]), 
                                      cvRound(pars->circles[j][1])
                                    );
        int radius = cvRound( pars->circles[j][2] );
        cv::circle( drawed, center, radius, circle_color, thickness );
    }
        
    cv::imshow( pars->window, drawed );
    
    return;
}

// ##################################################################



// ######################### LIBRARY FUNCTIONS ######################

    // function to compute the edges using Canny detector
    cv::Mat StreetFeaturesUtils::DetectEdges( cv::Mat& image ) {
        
        // create trackbar to select best parameters
        std::string window_name = "Set Canny parameters";       
        cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
        
        // class object to be passed to callback
        TrackbarParams pars;
        pars.window = window_name;
        pars.image  = image.clone();
        
        // convert image to grayscale
        cv::cvtColor(image, pars.filtered, cv::COLOR_BGR2GRAY);

        // creating trackbars
        int init_low = 250, init_high = 800;  // initial values for trackbars
        int max_th = 1000;                    // max value for thresholds
        
        pars.tb_names = { "Low Threshold : ", "High Threshold: " };
        
        cv::createTrackbar( pars.tb_names[0], window_name, &init_low , max_th, onCannyTrackbar, (void*)&pars );
        cv::createTrackbar( pars.tb_names[1], window_name, &init_high, max_th, onCannyTrackbar, (void*)&pars );
        
        onCannyTrackbar(0, &pars);
        
        cv::waitKey(0); 
        
        cv::destroyWindow( window_name );
        
        return pars.edges.clone();

    }

    
    // function to compute the lines from the edges using Hough transform
    std::vector<cv::Vec3f> StreetFeaturesUtils::RetrieveLines( cv::Mat& image, cv::Mat& edges ) {
        
        std::vector<cv::Vec3f> lines;
        
        // create trackbar to select best parameters
        std::string window_name = "Set Hough transform parameters";       
        cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
        
        // class object to be passed to callback
        TrackbarParams pars;
        pars.window = window_name;
        pars.image  = image.clone();
        pars.edges  = edges.clone();

        // creating trackbars
        int init_rho = 1, init_theta = 2, th_count = 150, init_min = 0, init_max = 180;  // initial values for trackbars
        
        pars.tb_names = { "Rho res. [pixels]: ", "Theta res. [deg] : ", "Count threshold  : ",
                          "Min. theta [deg] : ", "Max. theta [deg] : "
                        };
            
        cv::createTrackbar( pars.tb_names[0], window_name, &init_rho  ,  20, onLinesTrackbar, (void*)&pars );
        cv::setTrackbarMin( pars.tb_names[0], window_name, 1 );
        cv::createTrackbar( pars.tb_names[1], window_name, &init_theta,  20, onLinesTrackbar, (void*)&pars );
        cv::setTrackbarMin( pars.tb_names[1], window_name, 1 );
        cv::createTrackbar( pars.tb_names[2], window_name, &th_count  , 800, onLinesTrackbar, (void*)&pars );
        cv::setTrackbarMin( pars.tb_names[2], window_name, 5 );
        
        cv::createTrackbar( pars.tb_names[3], window_name, &init_min  , 180, onLinesTrackbar, (void*)&pars );
        cv::createTrackbar( pars.tb_names[4], window_name, &init_max  , 180, onLinesTrackbar, (void*)&pars );
        
        onLinesTrackbar(0, &pars);
        
        cv::waitKey(0);   
        
        cv::destroyWindow( window_name );
        
        return pars.lines;
        
    }
    
    
    // function to detect round signs detection with Hough transform
    std::vector<cv::Vec4f> StreetFeaturesUtils::RetrieveCircles( cv::Mat& image, double sigma ) {
        
        // create trackbar to select best parameters
        std::string window_name = "Set HoughCircles parameters";       
        cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
        
        // class object to be passed to callback
        TrackbarParams pars;
        pars.window = window_name;
        pars.image  = image.clone();
        
        // filtering 
        pars.filtered = StreetFeaturesUtils::ApplyFilter( image, sigma );
        
        // convert image to grayscale
        cv::cvtColor(pars.filtered, pars.filtered, cv::COLOR_BGR2GRAY);
        
        // creating trackbars
        int init_minDist = 20, init_th_high = 100, init_th_acc = 100, init_minR = 1, init_maxR = 100;  // initial values for trackbars
        
        pars.tb_names = { "Min distance   : ",
                          "Canny high th. : ",
                          "Acc. count th. : ",
                          "Min radius     : ",
                          "Max radius     : "
                        };

        cv::createTrackbar( pars.tb_names[0], window_name, &init_minDist, 1000, onCirclesTrackbar, (void*)&pars );
        cv::setTrackbarMin( pars.tb_names[0], window_name, 1 );
        cv::createTrackbar( pars.tb_names[1], window_name, &init_th_high, 500 , onCirclesTrackbar, (void*)&pars );
        cv::setTrackbarMin( pars.tb_names[1], window_name, 1 );
        cv::createTrackbar( pars.tb_names[2], window_name, &init_th_acc , 500 , onCirclesTrackbar, (void*)&pars );
        cv::setTrackbarMin( pars.tb_names[2], window_name, 1 );
        cv::createTrackbar( pars.tb_names[3], window_name, &init_minR   , 500 , onCirclesTrackbar, (void*)&pars );
        cv::setTrackbarMin( pars.tb_names[3], window_name, 1 );
        cv::createTrackbar( pars.tb_names[4], window_name, &init_maxR   , 500 , onCirclesTrackbar, (void*)&pars );
        cv::setTrackbarMin( pars.tb_names[4], window_name, 1 );
        
        onCirclesTrackbar(0, &pars);
        
        cv::waitKey(0);     

        cv::destroyWindow( window_name );        
        
        return pars.circles;
    
    }
    
    // function to plot the image with detected street features
    void StreetFeaturesUtils::SaveResult( cv::Mat& input_img, 
                                          std::vector<cv::Vec3f> lines, 
                                          std::vector<cv::Vec4f> circles, 
                                          std::string outname,
                                          std::string plot_option
                                        ) 
    {
        
        cv::Mat output = input_img.clone();
        
        if (plot_option == "region" or plot_option == "both") {   // add street lane region
            std::vector<cv::Point> poly_pts = StreetFeaturesUtils::ComputePolyVertices( lines,
                                                                                        input_img.size()
                                                                                      ); 
            if (poly_pts.size() > 2) {
                cv::fillConvexPoly( output, poly_pts, region_color );
            }
        }
        if (plot_option == "lines" or plot_option == "both") {   // add detected lines
            output = StreetFeaturesUtils::DrawLines( lines, output );
        }
            
        // add circles
        int thickness = -1; // -1 means fill        
        for( int j=0; j<circles.size(); j++) {
            cv::Point center = cv::Point( cvRound(circles[j][0]), 
                                          cvRound(circles[j][1])
                                        );
            int radius = cvRound( circles[j][2] );
            cv::circle( output, center, radius, circle_color, thickness );
        }
        
        cv::imshow( "Final result", output );
        cv::imwrite( outname, output );
        
        cv::waitKey(0);
        
        return;
    }
    
    

    // function to overlap red lines to the image
    cv::Mat StreetFeaturesUtils::DrawLines( std::vector<cv::Vec3f>& lines, 
                                            cv::Mat& img
                                          ) 
    {
        cv::Mat result = img.clone();
        int thickness = 2;
        
        // draw all the lines
        for( int i = 0; i < lines.size(); i++ ) {
            double rho = lines[i][0], theta = lines[i][1];
            cv::Point pt1, pt2;
            double ct = std::cos(theta), st = std::sin(theta);
            double x0 = ct*rho, y0 = st*rho;
            pt1.x = cvRound(x0 + 3000*(-st));
            pt1.y = cvRound(y0 + 3000*(ct));
            pt2.x = cvRound(x0 - 3000*(-st));
            pt2.y = cvRound(y0 - 3000*(ct));
            cv::line( result, pt1, pt2, line_color, thickness );
        }
        
        return result.clone();
    }
    
    
    // function to filter the image before circles detection
    cv::Mat StreetFeaturesUtils::ApplyFilter( cv::Mat& image, 
                                              double sigma
                                            )
    {
        cv::Mat result;
        
        if (not(sigma < 1)) {
            cv::GaussianBlur( image, result, cv::Size(0,0), sigma );
        }
        else {  // copy without filter
            result = image.clone();
        }
        
        return result;
    }
    
    
    // function to compute polygon between detected lines
    std::vector<cv::Point> StreetFeaturesUtils::ComputePolyVertices( std::vector<cv::Vec3f>& lines,
                                                                     cv::Size img_size
                                                                   )
    {
        std::vector<cv::Point> points;
        if (lines.size() < 2) {
            std::cout << "ERROR: needed at least 2 lines for polygon vertices computation, ";
            std::cout << "but " << lines.size() << " were provided.\n";
            std::cout << "   Skipping polygon vertices computation.\n";
            
            return points;
        }        
        
        //slopes 
        float m1 = - std::cos(lines[0][1]) / std::sin(lines[0][1]);
        float m2 = - std::cos(lines[1][1]) / std::sin(lines[1][1]);
        
        //intercepts
        float q1 = lines[0][0] / std::sin(lines[0][1]);
        float q2 = lines[1][0] / std::sin(lines[1][1]);  
        
        // points
        cv::Point cross, top_1, top_2, bot_1, bot_2;
        
        //bottom points
        bot_1.y = img_size.height -1 ;
        bot_2.y = img_size.height -1 ;        
        bot_1.x = cvRound( (q1 - (img_size.height -1)) / (-m1) );
        bot_2.x = cvRound( (q2 - (img_size.height -1)) / (-m2) );
        
        //cross point
        if (std::abs(m1 - m2) < 1e-8) { //lines are "almost" parallel 
            top_1.y = 0;
            top_2.y = 0;
            
            top_1.x = cvRound( - q1 / m1 );
            top_2.x = cvRound( - q2 / m2 );
            
            points.push_back(bot_1);
            points.push_back(top_1);
            points.push_back(top_2);
            points.push_back(bot_2);
        }
        else {   // good cross point
            cross.x = cvRound( (q2 - q1) / (m1 - m2) );
            cross.y = cvRound( m1 * (q2 - q1) / (m1 - m2) + q1 );
            
            points.push_back(bot_1);
            points.push_back(cross);
            points.push_back(bot_2);
        }   
        
        return points;
    }
                

