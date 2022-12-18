/*#-------------------------------------------------
#
#     Images comparison library with openCV
#
#    by AbsurdePhoton - www.absurdephoton.fr
#
#                v1.0 - 2022/12/12
#
#   - OpenCV's and custom hashes algorithms
#   - Image features and homography
#   - Compare image palettes
#   - Image quality
#   - Image comparison with DNN
#
#   uses OpenCV Contrib (features and DNN support)
#
#-------------------------------------------------*/

#ifndef IMAGECOMPARE_H
#define IMAGECOMPARE_H

#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/img_hash.hpp"
#include "opencv2/quality.hpp"

#include "image-transform.h"
#include "color-spaces.h"
#include "string-utils.h"
#include "contours.h"

#include <QCryptographicHash>
#include <QString>

#include <bitset>

// list of image matching algorithms available
enum imageSimilarityAlgorithm   {img_similarity_checksum,
                                 img_similarity_aHash, img_similarity_pHash, img_similarity_dHash, img_similarity_idHash, img_similarity_visHash,
                                 img_similarity_block_mean, img_similarity_color_moments, img_similarity_marr_hildreth, img_similarity_radial_variance,
                                 img_similarity_dominant_colors, // special algorithm - do not use image hash functions for these ones, use direct functions instead !
                                 img_similarity_features, img_similarity_homography, // special algorithm - do not use image hash functions for these ones, use direct functions instead !
                                 img_similarity_dnn_classify, // special algorithm - do not use image hash functions for these ones, use direct functions instead !
                                 img_similarity_count // always leave this one here
                   }; // image hash types


//// Image Hashes
cv::Mat ImageHash(const cv::Mat &source, const imageSimilarityAlgorithm &similarityAlgorithm); // return image hash as cv::Mat
float ImageHashCompare(const cv::Mat &val1, const cv::Mat &val2, const imageSimilarityAlgorithm &similarityAlgorithm); // compare 2 image hashes, return return % of similarity (NOT for special algorithms)
std::string Hash8U2String(const cv::Mat &source); // return a hex string from CV_8U hash
std::string HashChecksum2String(const cv::Mat &source); // return a hex string from checksum hash
//// Image Features and Homography
void ComputeImageDescriptors(const cv::Mat &source, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors,
                             const bool &resize=true, const int &size=256, const int &maxFeatures=250); // compute image features descriptors with ORB and BEBLID
float CompareImagesDescriptors(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                               const float &matchPercent=0.8f, const int &maxFeatures=250); // compare two images pre-computed descriptors -> % of similarity
cv::Mat DrawImageMatchesFromFeatures(const cv::Mat &im1, const cv::Mat &im2,
                                     std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                                     cv::Mat &descriptors1, cv::Mat &descriptors2,
                                     const float &matchPercent=0.8f, const int &maxFeatures=250); // compare two images -> get double image with best matches + homography
bool IsHomographyValid(const cv::Mat &HM, const bool &testRotation, const bool &testScale, const bool &testPerspective); // tells if a homography matrix is "good"
cv::Mat GetHomographyFromImagesFeatures(const cv::Mat &im1, const cv::Mat &im2,
                                        std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                                        cv::Mat &descriptors1, cv::Mat &descriptors2,
                                        std::vector<cv::Point2f> &goodPoints1, std::vector<cv::Point2f> &goodPoints2,
                                        float &score, const bool &resize=true, const int &size=256,
                                        const bool &toGray=true, const float &matchPercent=0.8f, const int &maxFeatures=250); // compare two images -> get homography matrix
//// Image Quality
float CompareImagesSSIM(const cv::Mat &im1, const cv::Mat &im2); // compare images with SSIM algorithm

//// Image color
float CompareImagesDominantColorsFromEigen(const std::vector<cv::Vec3d> &palette1, const std::vector<cv::Vec3d> &palette2); // mean color distance between palettes obtained from "Eigen" algorithm in dominant-colors.h
float CompareImagesDominantColorsFromSectoredMeans(const std::vector<std::vector<int>> &palette1, const std::vector<std::vector<int>> &palette2,
                                                   const float &percentage=0.05f, const double &threshold=2.0); // mean color distance between palettes obtained from "Sectored Means" algorithm in dominant-colors.h

//// DNN
void DNNPrepare(cv::dnn::Net &net, const std::string &model, const std::string &proto=""); // prepare DNN before hashing
cv::Mat DNNHash(const cv::Mat &image, cv::dnn::Net &net, const int &size=224, const cv::Scalar &mean=cv::Scalar(117, 117, 117), const int &nbValues=16); // get hash from model
float DNNCompare(const cv::Mat &output1, const cv::Mat &output2); // compare 2 results from DNNHash


#endif // IMAGECOMPARE_H
