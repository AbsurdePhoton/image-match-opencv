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

#include "image-compare.h"


///////////////////////////////////////////////////////////
//// Hashes : OpenCV's and custom
///////////////////////////////////////////////////////////

    // algorithms are explained here : https://hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
    // and there for advanced info : https://www.hackerfactor.com/blog/?/archives/529-Kind-of-Like-That.html

cv::Mat ImageChecksum(const cv::Mat &source, const QCryptographicHash::Algorithm &similarityMethod) // compute checksum (choose MD5, SHA1, etc) from image DATA (NOT file data)
{
    if (source.empty()) { // empty image
        return cv::Mat(); // -> empty result !
    }

    QByteArray ImgByteI((char*)source.data, (int)source.step[0] * source.rows); // cast image data from cv::Mat to list of chars
    QByteArray resultQ = QCryptographicHash::hash(ImgByteI, similarityMethod); // get hash, for example Quse CryptographicHash::Md5

    std::string hash = QString(resultQ.toHex()).toUtf8().constData(); // get hash in hex form

    // transform hex values in string to bytes in cv::Mat
    cv::Mat checksum = cv::Mat(1, hash.length(), CV_8UC1); // hash has a variable length, ontained from hash algo (MD5, SHA1, etc)
    uchar* checksumP = checksum.ptr<uchar>(0); // pointer to hash
    for (int i = 0; i < int(hash.length()); i++) // parse all values
        checksumP[i] = hash[i]; // copy value

    return checksum;
}

cv::Mat DifferenceHash(const cv::Mat &source)
    // see : https://www.hackerfactor.com/blog/?/archives/529-Kind-of-Like-That.html
    // a clever and efficient hashing algorithm !
    // works on a very tiny version of original image
    // VERY FAST
{
    // step 1 : resize image to 9x8
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(8, 9), 0, 0, cv::INTER_LINEAR_EXACT); // LINEAR_EXACT gives the best results

    // step 2 : reduce colors (to gray)
    cv::Mat gray;
    if(source.channels() > 1)
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    else
        gray = resized;

    // step 3 : compare values in gray image, create hash
    cv::Mat hash = cv::Mat::zeros(1, 8, CV_8U); // final result
    uchar* hashP = hash.ptr<uchar>(0); // pointer to it
    #pragma omp critical
    {
        std::bitset<8> bits; // temp bitset

        for (int j = 0; j < 8; j++) { // for each line
            const uchar *grayP = gray.ptr<uchar>(j); // pointer to line in gray image
            for (int i = 0; i < 8; i++) { // for each value of the line
                bits[i] = (grayP[i + 1] < grayP[i]); // set bit to 1 if left pixel is brighter, 0 if not
            }
            hashP[j] = static_cast<uchar>(bits.to_ulong()); // set hash value for this line from bits
        }
    }

    return hash;
}

cv::Mat ImportantDifferenceHash(const cv::Mat &source)
    // derived from DifferenceHash, which only scans the image horizontally : it can miss vertical features
    // this one combines horizontal AND vertical values - my own design
    // comparing hashes gets less matches than DifferenceHash BUT gets some that DifferenceHash missed...
    // it is a little slower than DifferenceHash but still VERY FAST
{
    // step 1 : resize to 9x9
    cv::Mat resized;
    cv::resize(source, resized, cv::Size(9, 9), 0, 0, cv::INTER_LINEAR_EXACT); // LINEAR_EXACT gives the best results

    // step 2 : reduce colors (to gray)
    cv::Mat grayH;
    if(source.channels() > 1)
        cv::cvtColor(resized, grayH, cv::COLOR_BGR2GRAY);
    else
        grayH = resized;
    // transpose gray image for vertical scan
    cv::Mat grayV;
    cv::transpose(grayH, grayV);

    // step 3 : compare values in grayH image, create hash
    cv::Mat hash = cv::Mat::zeros(1, 16, CV_8U); // final result
    #pragma omp critical
    {
        uchar* hashP = hash.ptr<uchar>(0); // pointer to it
        std::bitset<8> bitsH; // bits for lines
        std::bitset<8> bitsV; // bits for columns

        for (int j = 0; j < 8; j++) { // for each line
            const uchar *grayHP = grayH.ptr<uchar>(j); // pointer to line in grayH image
            const uchar *grayVP = grayV.ptr<uchar>(j); // pointer to line in grayV image
            for (int i = 0; i < 8; i++) { // for each column of the line
                bitsH[i] = (grayHP[i + 1] < grayHP[i]); // set bit to 1 if left pixel is brighter, 0 if not
                bitsV[i] = (grayVP[i + 1] < grayVP[i]); // set bit to 1 if left pixel is brighter, 0 if not - in fact it is vertical because grayV is the transpose of grayH !
            }
            hashP[j] = static_cast<uchar>(bitsH.to_ulong()); // set hash value for this line
            hashP[j + 8] = static_cast<uchar>(bitsV.to_ulong()); // set hash value for this column
        }
    }

    return hash;
}

cv::Mat ColorMomentsHash(const cv::Mat &source) // reimplementation of OpenCV's function, looks like there's a leak in it
    // needs a BGR image
{
    cv::Mat img;
    source.copyTo(img);

    // blur
    cv::GaussianBlur(img, img, cv::Size(3,3), 0, 0);
    // results
    cv::Mat result = cv::Mat::zeros(1, 42, CV_64F);
    double *resultP = NULL;
    // HSV part
    cv::Mat colorSpace;
    std::vector<cv::Mat> channels;
    cv::cvtColor(img, colorSpace, cv::COLOR_BGR2HSV); // convert image to HSV
    cv::split(colorSpace, channels); // split its channels
    // HSV moments
    resultP = result.ptr<double>(0);
    int count = 0; // important : advances as results are added
    for (int channel = 0; channel < 3; channel++) { // for each channel
        cv::Mat Hu;
        cv::HuMoments(cv::moments(channels[channel]), Hu); // compute moments
        double *HuP = Hu.ptr<double>(0);
        for (int n = 0; n < Hu.cols; n++) { // copy result to hash
            resultP[count] = HuP[n];
            count++;
        }
    }
    // YCrCb part - the same as before, but with YCrCb color space
    cv::cvtColor(img, colorSpace, cv::COLOR_BGR2YCrCb);
    channels.clear();
    cv::split(colorSpace, channels);
    // YCrCb moments
    for (int channel = 0; channel < 3; channel++) {
        cv::Mat Hu;
        cv::HuMoments(cv::moments(channels[channel]), Hu);
        double *HuP = Hu.ptr<double>(0);
        for (int n = 0; n < Hu.cols; n++) {
            resultP[count] = HuP[n];
            count++;
        }
    }

    return result;
}

cv::Mat ImageHash(const cv::Mat &source, const imageSimilarityAlgorithm &similarityAlgorithm) // return image hash as cv::Mat
    // not working for features and homography and DNN and dominant colors !
{
    if (source.empty()) {
        return cv::Mat();
    }

    cv::Mat hash;

    switch (similarityAlgorithm) {
        case img_similarity_checksum: {
            hash = ImageChecksum(source, QCryptographicHash::Md5);
            break;
        }
        case img_similarity_aHash: {
            cv::Ptr<cv::img_hash::AverageHash> h = cv::img_hash::AverageHash::create();
            h->compute(source, hash);
            break;
        }
        case img_similarity_pHash: {
            cv::Ptr<cv::img_hash::PHash> h = cv::img_hash::PHash::create();
            h->compute(source, hash);
            break;
        }
        case img_similarity_dHash: {
            hash = DifferenceHash(source);
            break;
        }
        case img_similarity_idHash: {
            hash = ImportantDifferenceHash(source);
            break;
        }
        case img_similarity_visHash: {
            hash = cv::Mat();
            break;
        }
        case img_similarity_block_mean: {
            cv::Ptr<cv::img_hash::BlockMeanHash> h = cv::img_hash::BlockMeanHash::create();
            h->compute(source, hash);
            break;
        }
        case img_similarity_color_moments: {
            hash = ColorMomentsHash(source);
            break;
        }
        case img_similarity_marr_hildreth: {
            cv::Ptr<cv::img_hash::MarrHildrethHash> h = cv::img_hash::MarrHildrethHash::create();
            h->compute(source, hash);
            break;
        }
        case img_similarity_radial_variance: {
            cv::Ptr<cv::img_hash::RadialVarianceHash> h = cv::img_hash::RadialVarianceHash::create();
            h->compute(source, hash);
            break;
        }
    }

    return hash;
}

///////////////////////////////////////////////////////////
//// Compare Hashes
///////////////////////////////////////////////////////////

float ImageHashCompare(const cv::Mat &val1, const cv::Mat &val2, const imageSimilarityAlgorithm &similarityAlgorithm) // compare 2 image hashes, return % of similarity (NOT for special algorithms)
    // not working for features and homography and DNN and dominant colors !
{
    if ((val1.empty()) or (val2.empty())) {
        return 0;
    }

    float dif = 0;

    switch (similarityAlgorithm) {
        case img_similarity_checksum: { // 16 uchar = 128 bits
            dif = (128.0f - cv::norm(val1, val2, cv::NORM_HAMMING)) / 1.28f;
            break;
        }
        case img_similarity_aHash: { // 8 uchar = 64 bits
            cv::Ptr<cv::img_hash::AverageHash> h = cv::img_hash::AverageHash::create();
            dif = (64.0f - h->compare(val1, val2)) / 64.0f;
            break;
        }
        case img_similarity_pHash: { // 8 uchar = 64 bits
            cv::Ptr<cv::img_hash::PHash> h = cv::img_hash::PHash::create();
            dif = (64.0f - h->compare(val1, val2)) / 0.64f;
            break;
        }
        case img_similarity_dHash: { // 8 uchar = 64 bits
            dif = (64.0f - cv::norm(val1, val2, cv::NORM_HAMMING)) / 0.64f;
            break;
        }
        case img_similarity_idHash: { // 16 uchar = 128 bits
            dif = (128.0f - cv::norm(val1, val2, cv::NORM_HAMMING)) / 1.28f;
            break;
        }
        case img_similarity_visHash: {
            dif = 0;
            break;
        }
        case img_similarity_block_mean: { // 121 uchar = 968 bits
            cv::Ptr<cv::img_hash::BlockMeanHash> h = cv::img_hash::BlockMeanHash::create();
            dif = (((968.0f - h->compare(val1, val2)) / 9.68f) -74.0f) * 3.95f - 2.7f; // result is between 0..100
            break;
        }
        case img_similarity_color_moments: { // ? 42 double, norm_l2 * 10000 !
            dif = (1.0f - cv::norm(val1, val2, cv::NORM_L2) * 20.0f) * 100.0f;
            break;
        }
        case img_similarity_marr_hildreth: { // 72 uchar = 64 bits
            cv::Ptr<cv::img_hash::MarrHildrethHash> h = cv::img_hash::MarrHildrethHash::create();
            dif = (576.0f - h->compare(val1, val2)) / 5.76f;
            break;
        }
        case img_similarity_radial_variance: { // 40 uchar = 320 bits
            cv::Ptr<cv::img_hash::RadialVarianceHash> h = cv::img_hash::RadialVarianceHash::create();
            dif = h->compare(val1, val2) * 100.0f;
            break;
        }
    }

    return dif;
}

///////////////////////////////////////////////////////////
//// Hash utils
///////////////////////////////////////////////////////////

std::string Hash8U2String(const cv::Mat &source) // return a hex string from CV_8U hash
{
    if (source.empty())
        return "";

    std::string out = "";
    const uchar* sourceP = source.ptr<uchar>(0);

    int total = source.cols * source.rows * source.channels();
    for (int n = 0; n < total; n++)
        out += stringutils::int2Hex(sourceP[n], 2);

    return out;
}

std::string HashChecksum2String(const cv::Mat &source) // return a hex string from checksum hash
{
    if (source.empty())
        return "";

    int total = source.total();
    std::vector<uchar> data(source.ptr(), source.ptr() + total);
    std::string result(data.begin(), data.end());

    return result;
}

///////////////////////////////////////////////////////////
//// Image Features
///////////////////////////////////////////////////////////

void ComputeImageDescriptors(const cv::Mat &source, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, const bool &resize, const int &size, const int &maxFeatures) // compute image features descriptors with ORB and BEBLID
    // resize image to size x size before computing if "resize" is true
{
    if (source.empty())
        return;

    // Convert image to grayscale
    cv::Mat gray;
    if (!resize)
        gray = source;
    else
        gray = ResizeImageAspectRatio(source, cv::Size(size, size));

    // compute keypoints with ORB + HARRIS
    keypoints.clear();
    keypoints.reserve(maxFeatures);
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(maxFeatures, 1.2, 4, 8, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20); // features options
    detector->detect(gray, keypoints);

    if (keypoints.empty())
        return;

    // compute descriptors with BEBLID
    cv::Ptr<cv::xfeatures2d::BEBLID> descriptor = cv::xfeatures2d::BEBLID::create(1.0f, cv::xfeatures2d::BEBLID::SIZE_512_BITS); // 1.0 is the recommended value for ORB
    descriptor->compute(gray, keypoints, descriptors);
}

float CompareImagesDescriptors(const cv::Mat &descriptors1, const cv::Mat &descriptors2, const float &matchPercent, const int &maxFeatures) // compare two images pre-computed descriptors -> % of similarity
{
    if ((descriptors1.empty()) or (descriptors2.empty()))
        return 0;

    // find descriptors matches with brute force method and knn
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    knnMatches.reserve(maxFeatures);
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    if (knnMatches.empty())
        return 0;

    // filter "good" matches with Lowe's ratio test
    int goodMatches = 0;
    for (int i = 0; i < int(knnMatches.size()); i++) {
        if (knnMatches[i][0].distance < matchPercent * knnMatches[i][1].distance) // matchPercent is used here - author suggests 0.8 but tests show that 0.85 is a bit better
            goodMatches++; // count good matches
    }

    return float(goodMatches) / float(maxFeatures); // percentage is computed from initial number of features
}

cv::Mat DrawImageMatchesFromFeatures(const cv::Mat &im1, const cv::Mat &im2,
                                     std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                                     cv::Mat &descriptors1, cv::Mat &descriptors2,
                                     const float &matchPercent, const int &maxFeatures) // compare two images -> get double image with best matches + homography
{
    if ((im1.empty()) or (im2.empty()) or (keypoints1.empty()) or (keypoints2.empty()) or (descriptors1.empty()) or (descriptors2.empty())) // no keypoints or descriptors, need both for both images
        return cv::Mat(); // return empty image

    // find descriptors matches with brute force method and knn
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector< std::vector<cv::DMatch>> knnMatches;
    knnMatches.reserve(maxFeatures);
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // filter "good" matches with Lowe's ratio test
    std::vector<cv::DMatch> goodMatches; // good ones will be here
    goodMatches.reserve(maxFeatures);
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < matchPercent * knnMatches[i][1].distance)
            goodMatches.push_back(knnMatches[i][0]);
    }

    // draw matches
    cv::Mat imgMatches;
    cv::drawMatches(im1, keypoints1, im2, keypoints2, goodMatches, imgMatches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //// Homography part

    if (goodMatches.size() < 8) // homography needs at least 4 points, let's double it !
        return imgMatches; // return image as is with only matches

    std::vector<cv::Point2f> points1;
    points1.reserve(maxFeatures);
    std::vector<cv::Point2f> points2;
    points2.reserve(maxFeatures);

    // get keypoints from good matches
    for (int i = 0; i < int(goodMatches.size()); i++) {
        points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
        points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
    }

    // compute homography using RHO algorithm
    cv::Mat mask;
    cv::Mat H = cv::findHomography(points1, points2, cv::RHO); // compute homography

    // test homography quality and draw it if needed
    if (IsHomographyValid(H, true, true, true)) {
        // corners of image 1 ( the object to be "detected" )
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cv::Point(0,0);
        obj_corners[1] = cv::Point(im1.cols, 0);
        obj_corners[2] = cv::Point(im1.cols, im1.rows);
        obj_corners[3] = cv::Point(0, im1.rows);

        // corners of image 1 transformed by homography
        std::vector<cv::Point2f> scene_corners(4);
        cv::perspectiveTransform(obj_corners, scene_corners, H);

        // Draw lines between the corners (the mapped object in the scene - image_2 )
        cv::line(imgMatches, scene_corners[0] + cv::Point2f( im1.cols, 0), scene_corners[1] + cv::Point2f( im1.cols, 0), cv::Scalar(0, 0, 255), 4);
        cv::line(imgMatches, scene_corners[1] + cv::Point2f( im1.cols, 0), scene_corners[2] + cv::Point2f( im1.cols, 0), cv::Scalar(0, 0, 255), 4);
        cv::line(imgMatches, scene_corners[2] + cv::Point2f( im1.cols, 0), scene_corners[3] + cv::Point2f( im1.cols, 0), cv::Scalar(0, 0, 255), 4);
        cv::line(imgMatches, scene_corners[3] + cv::Point2f( im1.cols, 0), scene_corners[0] + cv::Point2f( im1.cols, 0), cv::Scalar(0, 0, 255), 4);
    }

    return imgMatches;
}

bool IsHomographyValid(const cv::Mat &H, const bool &testRotation, const bool &testScale, const bool &testPerspective) // tells if a homography is "good"
{
    if (H.empty()) // empty matrix is bad
        return false;

    if (testRotation) {
        const double det = H.at<double>(0, 0) * H.at<double>(1, 1) - H.at<double>(1, 0) * H.at<double>(0, 1); // test orientation preservation with rotation component
        if (det < 0) // determinant < 0 means rotation/orientation has changed
            return false;
    }

    if (testScale) {
        const double N1 = sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(1, 0) * H.at<double>(1, 0)); // test scale, horizontal
        if (N1 > 1.5 || N1 < 0.25) // upscale more than 2x or downscale less than x4 ?
            return false;

        const double N2 = sqrt(H.at<double>(0, 1) * H.at<double>(0, 1) + H.at<double>(1, 1) * H.at<double>(1, 1)); // test scale, vertical
        if (N2 > 1.5 || N2 < 0.25) // upscale more than 2x or downscale less than x4 ?
            return false;
    }

    if (testPerspective) {
        const double N3 = sqrt(H.at<double>(2, 0) * H.at<double>(2, 0) + H.at<double>(2, 1) * H.at<double>(2, 1)); // test skew
        if (N3 > 0.003) // very thin ?
            return false;
    }

    return true;
}

cv::Mat GetHomographyFromImagesFeatures(const cv::Mat &im1, const cv::Mat &im2,
                                        std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2,
                                        cv::Mat &descriptors1, cv::Mat &descriptors2,
                                        std::vector<cv::Point2f> &goodPoints1, std::vector<cv::Point2f> &goodPoints2,
                                        float &score, const bool &resize, const int &size, const bool &toGray, const float &matchPercent, const int &maxFeatures) // compare two images -> get homography matrix

    // keypoints and descriptors can be fed from ComputeImageDescriptors(), which is used anyway if both are empty
{
    score = 0;

    // resize images if needed
    cv::Mat resized1, resized2;
    if (resize) {
        resized1 = QualityResizeImageAspectRatio(im1, cv::Size(size, size));
        resized2 = QualityResizeImageAspectRatio(im2, cv::Size(size, size));
    }
    else {
        resized1 = im1;
        resized2 = im2;
    }

    // convert images to grayscale if needed
    cv::Mat im1Gray, im2Gray;
    if (toGray) {
        cv::cvtColor(resized1, im1Gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(resized2, im2Gray, cv::COLOR_BGR2GRAY);
    }
    else {
        im1Gray = im1;
        im2Gray = im2;
    }

    // compute keypoints
    if ((keypoints1.empty()) or (descriptors1.empty())) {
        keypoints1.clear();
        ComputeImageDescriptors(im1Gray, keypoints1, descriptors1, false, size, maxFeatures);
    }
    if ((keypoints2.empty()) or (descriptors2.empty())) {
        keypoints2.clear();
        ComputeImageDescriptors(im2Gray, keypoints2, descriptors2, false, size, maxFeatures);
    }

    if ((keypoints1.empty()) or (keypoints2.empty()) or (descriptors1.empty()) or (descriptors2.empty())) // no keypoints or descriptors found ?
        return cv::Mat(); // return empty H matrix if not found

    // find descriptors matches with brute force method and knn
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    knnMatches.reserve(maxFeatures);
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

    if (knnMatches.empty()) // no matches found ?
        return cv::Mat(); // return empty H matrix

    // filter "good" matches using Lowe's ratio test
    std::vector<cv::DMatch> goodMatches;
    goodMatches.reserve(maxFeatures);
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < matchPercent * knnMatches[i][1].distance)
            goodMatches.push_back(knnMatches[i][0]);
    }

    if (goodMatches.size() < 8) // homography theorically needs at least 4 points, in fact it must be at least twice that amount
        return cv::Mat(); // return empty H matrix

    // get keypoints from good matches
    goodPoints1.reserve(maxFeatures);
    goodPoints2.reserve(maxFeatures);
    for (int i = 0; i < int(goodMatches.size()); i++) {
        goodPoints1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
        goodPoints2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
    }

    // compute homography using RHO algorithm
    cv::Mat mask;
    cv::Mat H = cv::findHomography(goodPoints1, goodPoints2, cv::RHO);

    // compute score
    if (IsHomographyValid(H, true, true, true)) {
        // corners from image 1
        std::vector<cv::Point2f> im1_corners(4);
        im1_corners[0] = cv::Point(0,0);
        im1_corners[1] = cv::Point(im1.cols, 0);
        im1_corners[2] = cv::Point(im1.cols, im1.rows);
        im1_corners[3] = cv::Point(0, im1.rows);
        // corners from image 2
        std::vector<cv::Point2f> im2_corners(4);
        im2_corners[0] = cv::Point(0,0);
        im2_corners[1] = cv::Point(im2.cols, 0);
        im2_corners[2] = cv::Point(im2.cols, im2.rows);
        im2_corners[3] = cv::Point(0, im2.rows);
        // corners of image 1 transformed by homography
        std::vector<cv::Point2f> im1_corners_transformed;
        cv::perspectiveTransform(im1_corners, im1_corners_transformed, H);
        // compute area from intersection of image 2 corners and transformed image 1 corners
        int area;
        cv::Mat intersection = SimpleContoursIntersectionToMask2f(im2_corners, im1_corners_transformed, area);
        score = float(area) / float(im2.cols * im2.rows); // percentage of "covered" image 2 by "homography-transformed" image 1
        if (score > 1.0f) // transformed image 1 can be bigger than image 2
            score = 1.0f;
    }

    return H;
}

///////////////////////////////////////////////////////////
//// Image Quality
///////////////////////////////////////////////////////////

float CompareImagesSSIM(const cv::Mat &im1, const cv::Mat &im2) // compare images with SSIM algorithm
    // beware : sizes of images and number of channels MUST be the same !
    // return 0 if error
    // not very usable as is
{
    if ((im1.channels() != im2.channels()) or (im1.cols != im2.cols) or (im1.rows != im2.rows))
        return 0;

    cv::Scalar q = cv::quality::QualitySSIM::compute(im1, im2, cv::Mat())[0];

    float result = 0;
    for (int c = 0; c < im1.channels(); c++)
        result += q[c];

    return float(result) / float(im1.channels());
}

///////////////////////////////////////////////////////////
//// Image Color
///////////////////////////////////////////////////////////

float CompareImagesDominantColorsFromEigen(const std::vector<cv::Vec3d> &palette1, const std::vector<cv::Vec3d> &palette2) // mean color distance between palettes obtained from "Eigen" algorithm in dominant-colors.h
    // palettes must have the same size !
{
    if ((palette1.empty()) or (palette2.empty()) or (palette1.size() != palette2.size())) // palettes empty or don't have the same number of elements ?
        return 0;

    double sum = 0;
    for (int n = 0; n < int(palette1.size()); n++) // for each palette values from both images
        sum += EuclideanDistanceSpace(palette1[n][0], palette1[n][1], palette1[n][2], palette2[n][0], palette2[n][1], palette2[n][2]); // just compute the color distance between them (values should be in OKLAB color space)

    return float(sum) / float(palette1.size()); // return mean color distance
}

///////////////////////////////////////////////////////////
//// DNN
///////////////////////////////////////////////////////////

void DNNPrepare(cv::dnn::Net &net, const std::string &model, const std::string &proto) // prepare DNN before hashing : backend and load model file
    // model is the path to model file (Torch or Caffe) - ex : "../models/neural-style/vgg16.t7"
{
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    if (proto == "")
        net = cv::dnn::readNetFromTorch(model);
    else
        net = cv::dnn::readNetFromCaffe(proto, model);
}

cv::Mat DNNHash(const cv::Mat &image, cv::dnn::Net &net, const int &size, const cv::Scalar &mean, const int &nbValues) // get hash from model features
    // input image is BGR and optianally resized
    // model should return a list of features in float values
{
    // prepare image
    cv::Mat img;
    image.copyTo(img);
    img.convertTo(img, CV_32FC3); // image must be in decimal (float) values

    // compute blob
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0, cv::Size(size, size), mean, false, false, CV_32F); // create blob imput from image

    // convolution -> get output
    cv::Mat output;
    #pragma omp critical
    {
        net.setInput(blob); // use the blob
        output =  net.forward(""); // get features
    }

    // get n "top" values, return them to matrix
    cv::Mat result = cv::Mat::zeros(2, nbValues, CV_32S); // return matrix
    int* resultP = result.ptr<int>(0); // pointer to result
    float* outputP = output.ptr<float>(0); // pointer to DNN output
    for (int n = 0; n < nbValues; n++) { // we want a defined number of values
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, 0, &confidence, 0, &classIdPoint); // get max value in DNN output and its location in the matrix
        int classId = classIdPoint.x; // the class id is the position in the DNN output matrix
        resultP[n] = classId; // save it to results
        resultP[n + nbValues] = confidence * 100.0; // with its confidence value (percentage)
        outputP[classId] = -1; // we don't want to find this resut again
    }

    return result;
}

float DNNCompare(const cv::Mat &output1, const cv::Mat &output2) // compare 2 results from DNNHash
{
    int sub = 4; // number of "next" values to test

    int count = 0;
    for (int i = 0; i < output1.cols - sub; i++) { // parse all values from ids
        for (int j = i; j < i + sub; j++) { // parse all "sub" values
            if ((output1.at<int>(0, i) != 0) and ((output1.at<int>(0, i) == output2.at<int>(0, j)))) { // if values are the same
                count ++; // +1 score !
                break; // don't parse other sub-values
            }
        }
    }

    return float(count) / float(output1.cols - sub); // percentage of matches
}
