/*#-------------------------------------------------
#
#       Contours utils library with OpenCV
#
#    by AbsurdePhoton - www.absurdephoton.fr
#
#                v1.2 - 2022/12/12
#
#   - Lines coordinates and interpolation
#   - Contours conversions, copy, bounding rect
#   - Contours drawing and get mask from drawing
#   - Contours transforms :
#       * scale
#       * mirror
#       * rotate
#       * translation
#       * random warp
#
#-------------------------------------------------*/

#ifndef CONTOURS_H
#define CONTOURS_H

#include "opencv2/opencv.hpp"

#include <lib/image-transform.h>

enum lineType {line_horizontal, line_vertical, line_oblique};
enum splineType {spline_centripetal, spline_uniform, spline_cordal};

struct CubicPoly {
    float c0, c1, c2, c3;

    float eval(float t)
    {
        float t2 = t * t;
        float t3 = t2 * t;
        return c0 + c1 * t + c2 * t2 + c3 * t3;
    }
};

/////////////////// Lines Utils ///////////////////

cv::Point LineCoordinatesAddLength(const cv::Point &p1, const cv::Point &p2, const double &length, const double &addedLength); // get new point coordinates on a line from 1st point based on length
int LineCoefficients(const cv::Point &p1, const cv::Point &p2, double &m, double &p); // from two points get line equation using slope formula
double LineCoordinateFromX(const double &m, const double &p, const int &lineType, const double &x); // get line coordinate Y from X using slope formula
double LineCoordinateFromY(const double &m, const double &p, const int &lineType, const double &y); // get line coordinate X from Y using slope formula


/////////////////  Contours Utils /////////////////

cv::Rect ContoursBoundingRect(const std::vector<std::vector<cv::Point>> &contours); // find bounding rectangle of array of contours
std::vector<cv::Point> SimpleContourFloatToInt(const std::vector<cv::Point2f> contour); // convert simple floats contour to integer
std::vector<cv::Point> CopySimpleContour(const std::vector<cv::Point> contour); // copy simple contour
std::vector<std::vector<cv::Point>> CopyContours(const std::vector<std::vector<cv::Point>> contours); // copy contours
std::vector<std::vector<cv::Point>> ConvertSimpleContour(const std::vector<cv::Point> &contour); // some opencv functions need vector of vector
std::vector<std::vector<cv::Point>> ConvertSimpleContour2f(const std::vector<cv::Point2f> &contour); // some opencv functions need vector of vector

///////////////  Contours Drawing ////////////////

void DrawSimpleContour(cv::Mat &source, const std::vector<cv::Point> &contour,
                       const cv::Scalar &color, const int &thickness = 1, const int &lineType = cv::LINE_8,
                       const cv::Point &offset = cv::Point(0, 0)); // opencv drawcontours needs an array of contours
void DrawContours(cv::Mat &source, const std::vector<std::vector<cv::Point>> &contours,
                  const cv::Scalar &color, const int &thickness, const int &lineType,
                  const cv::Point &offset); // draw several contours using DrawSimpleContour
void ContoursToMask(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, const int &index,
                    cv::Mat &mask, cv::Rect & rect); // return a mask + bounding rectangle from contours (+ hierarchy)
void SimpleContourToMask(const std::vector<cv::Point> &contour, cv::Mat &mask, cv::Rect & rect); // return a mask + bounding rectangle from simple contour
cv::Mat SimpleContoursIntersectionToMask(const std::vector<cv::Point> &contour1, const std::vector<cv::Point> &contour2, int &area); // return a mask of 2 contours intersection
cv::Mat SimpleContoursIntersectionToMask2f(const std::vector<cv::Point2f> &contour1, const std::vector<cv::Point2f> &contour2, int &area); // return a mask of 2 contours intersection


/////////////////// Transforms ////////////////////

void ScaleSimpleContour(std::vector<cv::Point> &contour, const float &scaleX=1.0, const float &scaleY=0, const int &width=0, const int &height=0); // rescale a contour
void ScaleContours(std::vector<std::vector<cv::Point>> &contours, const float &scaleX, const float &scaleY=0, const int &width=0, const int &height=0); // rescale array of contours
void MultiplyContours(std::vector<std::vector<cv::Point>> &contours, const float &scale); // multiply array of contours by a scale

void MirrorSimpleContour(std::vector<cv::Point> &contour, const bool &horizontal, const bool &vertical); // mirror simple contour along vertical and/or horizontal axis
void MirrorContours(std::vector<std::vector<cv::Point> > &contours, const bool &horizontal, const bool &vertical); // mirror array of contours along vertical and/or horizontal axis

void RotateSimpleContour(std::vector<cv::Point> &contour, const float &angleRad, const cv::Point2f &center=cv::Point2f()); // rotate a contour around a center
void RotateContours(std::vector<std::vector<cv::Point>> &contours, const float &angleRad, const cv::Point2f &center=cv::Point2f()); // rotate array of contours around a center

void WarpSimpleContourHorizontal(std::vector<cv::Point> &contour, const int &typeH, const int &pixelsX); // warp a simple contour (horizontal)
void WarpSimpleContourVertical(std::vector<cv::Point> &contour, const int &typeH, const int &pixelsY); // warp a simple contour (vertical)
void WarpContoursHorizontal(std::vector<std::vector<cv::Point> > &contours, const int &typeH, const int &pixelsX); // warp array of contours (horizontal)
void WarpContoursVertical(std::vector<std::vector<cv::Point> > &contours, const int &typeH, const int &pixelsY); // warp array of contours (vertical)

void TranslateSimpleContour(std::vector<cv::Point> &contour, const int &shiftHorizontal, const int &shiftVertical); // translate simple contour along vertical and/or horizontal axis
void TranslateContours(std::vector<std::vector<cv::Point>> &contours, const int &shiftHorizontal, const int &shiftVertical); // translate contours along vertical and/or horizontal axis

/////////////////// Interpolation ////////////////////
std::vector<cv::Point> GetCurvedLineCatmullRom(const cv::Point &p0, const cv::Point &p1, const cv::Point &p2, const cv::Point &p3, const splineType &sType, const int &nbSegments); // interpolate a line from 4 points using non-uniform Catmull-Rom formula

#endif // CONTOURS_H
