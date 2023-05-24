// Computer Vision 2021 - Final project: Panoramic image construction using SIFT
//  
// Author: Michele Guadagnini - Mt. 1230663
//------------------------------------------------------------------------------------------------- 
//
// This program allows to build a panoramic image from a sequence of images.
// It uses Scale Invariant Feature Transform (SIFT) to detect keypoints and 
//  compute descriptors.
//  
// PARAMETERS: 
//   images_path :   path to the images, with names pattern. 
//   FoV         :   Field of View of the camera in degrees. 
//   ratio       :   parameter used to refine matches selection. 
//   output_name :   name to use for the final panoramic image. 
//   color_mode  :   if to use color [RGB] or gray-scale [GRAY]. 
//   equalization:   if to apply equalization to final image [yes] or not [no] 
// 
// Usage  :   ./panoramic "images_path"  FoV  ratio  output_name  color_mode  equalization
// 
// Example:   ./panoramic "dolomites/i%2d.png"  54  2.5  dolomites_pan.png   RGB  yes
//-------------------------------------------------------------------------------------------------

// include standard headers
#include <iostream>
#include <string>
#include <vector>

// include openCV
#include <opencv2/opencv.hpp>

// include custom
#include "panoramic_utils.h"


using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;


// help message (in case of wrong number of arguments)
void help() {
    
    cout << "HELP MESSAGE: " << endl;
    cout << "This program allows to build a panoramic image from a sequence of images.\n" ;
    cout << "It uses Scale Invariant Feature Transform (SIFT) to detect keypoints and \n compute descriptors.\n";
    cout << "\nPARAMETERS: \n";
    cout << "  images_path :   path to the images, with names pattern. \n";
    cout << "  FoV         :   Field of View of the camera in degrees. \n";
    cout << "  ratio       :   parameter used to refine matches selection. \n";
    cout << "  output_name :   name to use for the final panoramic image. \n";
    cout << "  color_mode  :   if to use color [RGB] or gray-scale [GRAY]. \n";
    cout << "  equalization:   if to apply equalization to final image [yes] or not [no] \n";
    
    cout << "\nUsage  :   ./panoramic \"images_path\"  FoV  ratio  output_name  color_mode  equalization" << endl;
    cout << "\nExample:   ./panoramic \"dolomites/i%2d.png\"  54  2.5  dolomites_pan.png   RGB  yes" << endl;
    
    return;
}

//###################################################################
//######            functions used for testing                 ######

void display_img_test(cv::Mat &img);

void display_keypoints_test(cv::Mat &img, vector<cv::KeyPoint> &kps);

void display_descriptor_test(cv::Mat& img, cv::Mat& desc);

void draw_matches_test( cv::Mat& img1, 
                        vector<cv::KeyPoint>& keypoints1, 
                        cv::Mat& img2, 
                        vector<cv::KeyPoint>& keypoints2, 
                        vector<cv::DMatch>& matches );

void trsl_transform_test( cv::Mat& img1, cv::Mat& img2, cv::Mat& transformation );

//###################################################################








// BEGIN

int main(int argc, char** argv) {  

//############################ INPUT ################################    

    cout << "Reading and checking arguments... " << endl;
    // check passed arguments
    if ( argc != (6 +1) )
    {
        cout << "  Wrong number of arguments ! \n" << endl;
        help();
        return 1;
    }
    // reading arguments
    const string folderpath = argv[1];
    const double FoV        = std::stod(argv[2]);
    const double ratio      = std::stod(argv[3]);
    const string outname    = argv[4];
    const string color_mode = argv[5];
    const string equalize   = argv[6];
    
    // load image sequence            
    cv::VideoCapture sequence(folderpath);
    if (!sequence.isOpened())
    {
        cerr << "  Failed to open Image Sequence!\n" << endl;
        return 1;
    }
    cout << "  Image Sequence successfully loaded." << endl;
       
    // storing images into a vector
    vector<cv::Mat> images;
    while (true) {
        cv::Mat buff;
        sequence >> buff;
        
        if (buff.empty()) {  break;  }
        
        images.push_back(buff.clone());
    }
    cout << "  The sequence contains: " << images.size() << " images.\n" << endl;
    
//################### SINGLE IMAGE OPERATIONS #######################    

    cout << "Single Image operations: applying projection and computing SIFT... " << endl;
    
    vector<vector<cv::KeyPoint>> SIFT_kps(images.size());       //vector of vectors for SIFT keypoints
    vector<cv::Mat> SIFT_descs(images.size());                  //vector for SIFT descriptors
    
    for (int ii=0; ii < images.size(); ii++) {
        
        if (color_mode == "RGB") {
            // project on the cylindrical surface in RGB
            images.at(ii)   = PanoramicUtils::cylindricalProjRGB( images.at(ii), FoV/2. ); 
        }
        else {
            // project on the cylindrical surface in GRAY-SCALE
            images.at(ii)   = PanoramicUtils::cylindricalProj( images.at(ii), FoV/2. );
        }
        
        // SIFT extraction and computation
        PanoramicUtils::SIFTkeypoints( images.at(ii), SIFT_kps.at(ii), SIFT_descs.at(ii) ); 
    }
    cout << "  SIFT descriptors computed. \n" << endl;
    
//     display_img_test(images.at(12));                               // TEST
//     display_keypoints_test(images.at(12), SIFT_kps.at(12));        // TEST
//     display_descriptor_test(images.at(12), SIFT_descs.at(12));     // TEST
    
//########## OPERATIONS ON COUPLE OF consecutive IMAGES #############    

    cout << "Operations on adjacent images: computing matches and translations... " << endl;
    
    vector<vector<cv::DMatch>> SIFT_mtchs(images.size()-1);     //vector of vectors for matches between adjacent images
    vector<cv::Mat>            transforms(images.size()-1);     //vector to store translations between adjacent images
    
    for (int jj=0; jj < images.size()-1; jj++) {
        
        // compute the matches between descriptors
        SIFT_mtchs.at(jj) = PanoramicUtils::SIFTmatches( SIFT_descs.at(jj), SIFT_descs.at(jj+1), ratio );
        
        // RANSAC to find translations
        transforms.at(jj) = PanoramicUtils::find_translation( SIFT_kps.at(jj), SIFT_kps.at(jj+1), SIFT_mtchs.at(jj));
    }
    cout << "  Translations between adjacent images computed. \n" << endl;
    
//     draw_matches_test( images.at(12),                                  //TEST
//                        SIFT_kps.at(12), 
//                        images.at(13),
//                        SIFT_kps.at(13), 
//                        SIFT_mtchs.at(12) );   
    
//     warp_transform_test( images.at(0), images.at(1), transforms.at(0) );    //TEST
//     trsl_transform_test( images.at(0), images.at(1), transforms.at(0) );    //TEST
    
//################# BUILD THE PANORAMIC IMAGE #######################

    cout << "Building the panoramic image... " << endl;
    
    cv::Mat final_pan = PanoramicUtils::build_panoramic( images, transforms );
    
    cv::Mat panoramic;
    if ((equalize == "yes") or (equalize == "YES") or (equalize == "Yes") 
                            or (equalize == "Y")   or (equalize == "y")) {
        // equalize the image
        cout << "  Equalizing the resulting image... " << endl;
        panoramic = PanoramicUtils::intensity_hist_EQ( final_pan );
    }
    else {
        panoramic = final_pan.clone();
    }
    
    // save/visualize the image
    cv::imshow("panoramic", panoramic);
    cv::imwrite(outname, panoramic);
    
    cout << "  Done. Waiting for key to be pressed... " << endl;
    cv::waitKey(0);
    
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

void display_keypoints_test(cv::Mat &img, vector<cv::KeyPoint> &kps) {
    
    cv::Mat output;
    cv::drawKeypoints(img, kps, output);
    cv::imshow("sift_keypoints_result", output);
    cv::waitKey(0);
    
    return;
}

void display_descriptor_test(cv::Mat& img, cv::Mat& desc) {
 
    cv::imshow("image", img);
    cv::waitKey(0);
    
    cv::imshow("descriptor", desc);
    cv::waitKey(0);
    
    return;
}

void draw_matches_test( cv::Mat& img1, 
                        vector<cv::KeyPoint>& keypoints1, 
                        cv::Mat& img2, 
                        vector<cv::KeyPoint>& keypoints2, 
                        vector<cv::DMatch>& matches ) 
{    
    cv::Mat img_matches;
    cv::drawMatches( img1, keypoints1, img2, keypoints2, matches, img_matches );

    cv::imshow("Matches", img_matches );
    cv::imwrite("matches.png", img_matches);
    cv::waitKey(0);
    
    return;
}

void trsl_transform_test( cv::Mat& img1, cv::Mat& img2, cv::Mat& transformation ) {    
    
    cv::Mat result;
    
    double trl = transformation.at<double>(0,2);
    int t_x = static_cast<int>(trl + 0.5);
    cv::Size new_size = cv::Size(img1.cols + t_x, img1.rows);
    
    cv::warpAffine(img2, result, transformation, new_size, cv::INTER_LINEAR);
    cv::Mat half(result, cv::Rect(0,0,img1.cols,img1.rows));
    img1.copyTo(half);
    
    cv::imshow("stitch", result );
    cv::imwrite("stitching.png", result);
    cv::waitKey(0);
    
    return;
}
