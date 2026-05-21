/*#-------------------------------------------------
#
#        Image utils library with OpenCV
#
#    by AbsurdePhoton - www.absurdephoton.fr
#
#                v2.5 - 2025/01/24
#
#   - Utilities :
#       * Range values utils for int and uchar
#       * Count number of colors (RGB unique values)
#       * Normalize to and from 1.0 <--> 255
#       * cv::Mat equality (not obvious !)
#       * Is the image a solid color block ?
#   - Conversions OpenCV <--> Qt (QPixmap and QImage)
#   - Copy and paste (also with alpha) + fast versions
#   - Alpha images :
#       * add alpha channel
#       * load alpha image to BGR + transparency grid
#       * split channels with adding alpha
#   - Image analysis, get :
#       * Gray histogram (option with mask)
#       * Color mean using histograms
#       * Color palette
#   - Save PNG image with transparency
#   - Save float image as tif image file
#   - image orientation and orientation comparison
#
#-------------------------------------------------*/


#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H


#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

#include <QImage>
#include <QPixmap>

/* cv::Mat characyeristics :

    * cv::Mat::depth() :
      corresponds to the type of the unit used
        (e.g. example.depth() == CV_64F)

        CV_8U   uchar (unsigned char)   8-bit
        CV_8S   signed char             8-bit
        CV_16U  short int unsigned      16-bit
        CV_16S  short int signed        16-bit
        CV_32S  int signed              32-bit
        CV_32F  float                   64-bit
        CV_64F  double                  64-bit

    * cv::Mat::channels() :
      number of channels
        (e.g. CV_8UC3 is a BGR image : B is the blue channel, etc)

    * cv::Mat::type() :
      is the combination of depth() and channels() - use it to copy the type of another matrix or test its value
        (e.g. example.type() == CV_8UC1)

                C1 	C2 	C3 	C4
        CV_8U 	0 	8 	16 	24
        CV_8S 	1 	9 	17 	25
        CV_16U 	2 	10 	18 	26
        CV_16S 	3 	11 	19 	27
        CV_32S 	4 	12 	20 	28
        CV_32F 	5 	13 	21 	29
        CV_64F 	6 	14 	22 	30

    * cv::Mat::size() :
      dimensions : number of rows and columns

*/


enum orientationImage { orientation_portrait, orientation_landscape, orientation_square, orientation_error }; // image orientation
enum orientationImageComparison { orientation_comparison_same, orientation_comparison_different, orientation_comparison_square, orientation_comparison_error }; // image orientation comparison


//// General
int GetByteInRange(const int &byte); // get a byte in range [0..255]

inline double FastPow(double a, double b)
{
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;
    return u.d;
}

inline double FastPrecisePow(double a, double b) // should be much more precise with large b
    // doesn't work for negative exponents !!!
{
    // calculate approximation with fraction of the exponent
    int e = (int) b;
    union {
        double d;
        int x[2];
    } u = { a };
    u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;

    // exponentiation by squaring with the exponent's integer part
    // double r = u.d makes everything much slower, not sure why
    double r = 1.0;
    while (e) {
        if (e & 1)
            r *= a;
        a *= a;
        e >>= 1;
    }

    return r * u.d;
}

template <typename anyValue> bool IsValueInRange(anyValue param, anyValue min, anyValue max)
{
    return (param >= min) and (param <= max);
}

template <typename anyValue> anyValue GetValueInRange(anyValue param, anyValue min, anyValue max)
{
    if (param < min)
        return min;
    else if (param > max)
        return max;
    else
        return param;
}

//// Utils
int CountRGBUniqueValues(const cv::Mat &source); // count number of RGB colors in image
cv::Mat NormalizeImage(const cv::Mat & source, const bool &multiThread=false); // normalize [0..255] image to [0..1] - returns a CV_64F image
cv::Mat DeNormalizeImage(const cv::Mat & source, const bool &multiThread=false); // denormalize [0..1] image to [0..255] - needs a CV_64F input
bool MatEqual(const cv::Mat &one, const cv::Mat &two); // compare two Mat
bool IsImageSolidColor(const cv::Mat &source); // tells if BGR image consists of a unique color
void ClampMat(cv::Mat &source, const double &min, const double &max); // clamp matrix values between min and max
void SaveFloatMatToFile(const cv::Mat &m, const char *filename); // save a float cv::Mat to ascii file
cv::Mat ImageAnydepthToColor(const cv::Mat &source); // convert any image of any depth (even grayscale) to BGR

//// Conversions of cv::Mat to and from Qt images
cv::Mat QImage2Mat(const QImage &source); // convert QImage to Mat
cv::Mat QPixmap2Mat(const QPixmap &source); // convert QPixmap to Mat
QImage  Mat2QImage(const cv::Mat &source); // convert Mat to QImage
QPixmap Mat2QPixmap(const cv::Mat &source); // convert Mat to QPixmap
QPixmap Mat2QPixmapResized(const cv::Mat &source, const int &width, const int &height, const bool &smooth); // convert Mat to resized QPixmap
QImage  cvMatToQImage(const cv::Mat &source); // another implementation Mat type wise

//// Copy and paste
// general
cv::Rect RectangleFromImage(const cv::Mat &source, const cv::Rect &frame); // get a rectangle cropped to the image limits
// copy
cv::Mat CopyFromImage(const cv::Mat &source, const cv::Rect &frame); // copy part of image - if parts of frame are outside source image, non-copied pixels are black
// DEPRECATED : paste with optional transparency, old functions
void PasteImageGraySlow(cv::Mat &background, const cv::Mat &foreground,
                        const int &originX, const int &originY,
                        const bool &transparency=true, const uchar &transparentColor=0, const bool &multiThread=false); // copy gray image onto another with optional transparency
void PasteImageColorSlow(cv::Mat &destination, const cv::Mat &source,
                         const int &originX, const int &originY,
                         const bool &transparency=true, const cv::Vec3b &transparentColor=cv::Vec3b(0,0,0), const bool &multiThread=false); // copy BGR image onto another with optional transparency
// paste image with transparency, fast memcpy version
void PasteImageGrayTransparency(cv::Mat &background, const cv::Mat &foreground,
                                const int &originX=0, const int &originY=0,
                                const uchar &transparentColor=0, const bool &multiThread=false); // copy gray image onto another with transparency
void PasteImageColorTransparency(cv::Mat &background, const cv::Mat &foreground,
                                 const int &originX=0, const int &originY=0,
                                 const cv::Vec3b &transparentColor=cv::Vec3b(0, 0, 0), const bool &multiThread=false); // copy BGR image onto another with transparency
// DEPRECATED : paste image with alpha, old functions
void PasteImageAlphaSlow(cv::Mat &background, const cv::Mat &foreground, const cv::Point &pos=cv::Point(0, 0), const bool &multiThread=false); // paste alpha BGR image on non-alpha background
void PasteImagePlusAlphaSlow(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                             const cv::Point &pos=cv::Point(0, 0), const bool &multiThread=false); // paste (gray or color) + alpha image on non-alpha BGR background
void PasteImageAlphaOnAlphaSlow(cv::Mat &background, const cv::Mat &foreground,
                                const cv::Point &pos=cv::Point(0, 0), const bool &multiThread=false); // paste alpha BGR image on alpha BGR background
// paste image as is, no transparency or alpha - fast memcpy version
void PasteImageFast(cv::Mat &background, const cv::Mat &foreground, const int &originX=0, const int &originY=0); // copy a matrix onto another of any type - fast implementation
// paste image with alpha, fast memcpy version
void PasteImageAlphaFast(cv::Mat &background, const cv::Mat &foreground,
                         const cv::Point &pos=cv::Point(0, 0), const bool &multiThread=false); // paste alpha BGR image on non-alpha background - fast version
void PasteImageAlphaOnAlphaFast(cv::Mat &background, const cv::Mat &foreground,
                                const cv::Point &pos=cv::Point(0, 0), const bool &multiThread=false); // paste alpha BGR image on alpha BGR background - fast version
void PasteImagePlusAlphaFast(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                             const cv::Point &pos=cv::Point(0, 0), const bool &multiThread=false); // paste (gray or color) + alpha image on non-alpha BGR background - fast version
// special paste functions
void PasteImageGrayOnGrayOR(cv::Mat &background, const cv::Mat &foreground,
                            const cv::Point &pos=cv::Point(0, 0), const bool &multiThread=false); // paste gray image on gray background using a OR combination
int PasteImageGrayTransparencyWithCount(cv::Mat &background, const cv::Mat &foreground,
                                        const int &originX, const int &originY,
                                        const uchar &transparentColor, const bool &multiThread=false); // copy gray image onto another with transparency and count the updated pixels

//// Alpha channel
cv::Mat AddAlphaToImage(const cv::Mat &source, const cv::Mat &mask=cv::Mat()); // add alpha channel to BGR image - if mask is empty, transparency = black (0,0,0)
cv::Mat ImageGrid(const int &width, const int &height, const int &interval=32, const cv::Vec3b color1=cv::Vec3b(192,192,192), const cv::Vec3b color2=cv::Vec3b(127,127,127)); // get BGR image with colored blocks
cv::Mat ImageAlphaWithGrid(const cv::Mat &source, const int &interval=32, const cv::Vec3b color1=cv::Vec3b(192,192,192), const cv::Vec3b color2=cv::Vec3b(127,127,127)); // get BGR image from BGRA with colored blocks where there is transparency
void ImageToBGRplusAlpha(const cv::Mat &source, cv::Mat &dest, cv::Mat &alpha); // split an image with n channels to BGR + alpha
std::vector<cv::Mat> SplitImage(const cv::Mat source); // split image to channels

//// Analysis
std::vector<double> HistogramImageGray(const cv::Mat &source); // compute histogram of gray image
std::vector<double> HistogramImageGrayWithMask(const cv::Mat &source, const cv::Mat &mask); // compute histogram of gray image using a mask
cv::Vec3d MeanWeightedColor(const cv::Mat &source, const cv::Mat &mask); // use an histogram to get mean weighted color of an image area
double MeanWeightedGray(const cv::Mat &source, const cv::Mat &mask); // use an histogram to get mean weighted gray of an image area
cv::Mat CreatePaletteImageFromImage(const cv::Mat3b &source); // parse BGR image and create a one-line RGB palette image from all colors - super-fast !

//// Image files
void SavePNG(const std::string &filename, const cv::Mat &source, const bool &transparency); // save PNG with or without transparency
void SaveFloatImageAsTiff(const cv::Mat &source, const std::string &filename); // save CV_16F floating-point image to TIFF

//// MATLAB equivalents
enum Conv2DShape {
    conv2d_full,
    conv2d_same,
    conv2d_valid
};
cv::Mat conv2d(const cv::Mat &img, const cv::Mat &kernel, const Conv2DShape &shape); // equivalent to MATLAB conv2d
cv::Mat filter2(const cv::Mat &img, const cv::Mat &kernel); // equivalent to MATLAB filter2

//// Masks utils
cv::Rect FindMaskContentBoundingRect(const cv::Mat &source); // return the bounding rect of the content of a mask
cv::Mat AdjustMaskToRect(const cv::Mat &source, cv::Rect &rect); // ajust a gray mask to its content

//// DPI utils
cv::Size ComputeDPI(const double &diagonalUnits, const cv::Size &aspectRatio, const cv::Size &resolution); // compute DPI from diagonal and aspect ratio
cv::Size ComputeDPI(const cv::Size &sizeUnits, const cv::Size &resolution); // compute DPI from width anf height

//// Orientation of image
int GetImageOrientation(const cv::Mat &source); // get image orientation : portrait, landscape or square
int CompareImagesOrientation(const cv::Mat &source1, const cv::Mat &source2, const bool &ignoreSquares=true); // compare orientations of two images


#endif // IMAGEUTILS_H
