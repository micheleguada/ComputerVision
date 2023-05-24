// Computer Vision 2021 - Homework: Street lanes and round street signs detection (LAB 3)
//  
// Author: Michele Guadagnini - Mt. 1230663
//------------------------------------------------------------------------------------------------- 
//
// This program aims at detecting street lanes and round signs from pictures. It uses the Canny 
// edge detector and the Hough transform to detect lines and circles.
//  
// PARAMETERS: 
//   input_name     : name of the image file to process
//   output_name    : name of the output file 
//   plot_option    : type of output plot. The options are: 
//                      - "lines" : simply overlap detected lines to the image;
//                      - "region": color the region between the lines;
//                      - "both"  : does both the things above;
//   gaussian_sigma : sigma of the gaussian filter to apply before circles detection. If a value < 1 
//                      is passed, filtering will be skipped.
// 
// Compile: g++ -o street_features street_features_main.cpp street_features_utils.cpp `pkg-config --cflags --libs opencv`
//
// Usage  : ./street_features [input_name] [output_name] [plot_option] [gaussian_sigma]
// 
// Example: ./street_features  road7.jpg  road7_featured.jpg  region  5.
//------------------------------------------------------------------------------------------------- 


// include standard headers
#include <iostream>
#include <string>
#include <vector>

// include openCV
#include <opencv2/opencv.hpp>

// include custom
#include "street_features_utils.h"

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;


const int N_params = 4;


// help message (in case of wrong number of arguments)
void help() {
    
    cout << "HELP MESSAGE: " << endl;
    cout << "This program aims at detecting street lines and round signs from pictures.\n";
    cout << "It uses the Canny edge detector and the Hough transform to detect lines and circles.\n";
    cout << "All the parameters of the algorithms can be set by mean of trackbars\n";
    cout << "\nPARAMETERS: \n";
    cout << "   input_name     : name of the image file to process\n";
    cout << "   output_name    : name of the output file\n";
    cout << "   plot_option    : type of output plot. The options are:\n"; 
    cout << "                     - \"lines\" : simply overlap detected lines to the image;\n";
    cout << "                     - \"region\": color the region between the lines;\n";
    cout << "                     - \"both\"  : does both the things above;\n";    
    cout << "   gaussian_sigma : sigma of the gaussian filter to apply before circles detection.\n";
    cout << "                      If a value < 1 is passed, filtering will be skipped.\n";
    
    cout << "\nUsage  :  ./street_features [input_name] [output_name] [plot_option] [gaussian_sigma] " << endl;
    cout << "\nExample:  ./street_features  road7.jpg  road7_featured.jpg  region  5.  " << endl;
    
    return;
}

//###################################################################
//######            functions used for testing                 ######

void display_img_test(cv::Mat &img);


//###################################################################



// BEGIN

int main(int argc, char** argv) {  

    cout << "Reading and checking arguments... " << endl;
    // check passed arguments
    if ( argc != (N_params +1) )
    {
        cerr << "  Wrong number of arguments ! \n" << endl;
        help();
        return 1;
    }
    // reading arguments
    const string filename    = argv[1];
    const string outname     = argv[2];
    const string plot_option = argv[3];
    double gaussian_sigma    = std::atof(argv[4]);
    
//######################### LOAD IMAGE ##############################
    
    cv::Mat input_img = cv::imread(filename);
    
    if ( input_img.empty() )  // check for reading errors
    {
        cerr << "  Failed to open image !\n" << endl;
        return -1;
    }
    cout << "  Image successfully loaded." << endl;
    
//     // TEST
//     display_img_test(input_img);
    
//     // TEST filter
//     cv::Mat filter_test = StreetFeaturesUtils::ApplyFilter( input_img, gaussian_sigma );
//     display_img_test(filter_test);
    
//######################## LINES DETECTION ##########################
    
    cout << "\nPerforming lines detection... " << endl;
    
    //apply canny (with trackbars)
    cv::Mat edges = StreetFeaturesUtils::DetectEdges( input_img );  
    cout << "  Canny edge detector applied." << endl;
    
    //hough transform (with trackbars) 
    std::vector<cv::Vec3f> lines = StreetFeaturesUtils::RetrieveLines( input_img, edges );
    cout << "  Hough transform applied successfully." << endl;
    
    
//####################### CIRCLES DETECTION #########################
    
    cout << "\nPerforming round signs detection... " << endl;
    
    // hough circles (with trackbars)
    std::vector<cv::Vec4f> circles = StreetFeaturesUtils::RetrieveCircles( input_img, gaussian_sigma );
    cout << "  Hough transform applied successfully." << endl;
    
    
//###################### PRODUCE OUTPUT PICTURE #####################
    
    cout << "\nDrawing features on original image... " << endl;
    
    // produce and save output (draw lines and region, draw circles)
    StreetFeaturesUtils::SaveResult( input_img, lines, circles, outname, plot_option );
    cout << "  Processed image stored on disk. Exiting..." << endl;
    
    return 0;
}

//END



//###################################################################
//######            testing function definitions               ######

void display_img_test(cv::Mat &img) {
    
    cv::imshow("test", img);
    cv::waitKey(0);
    
    return;
}
       
