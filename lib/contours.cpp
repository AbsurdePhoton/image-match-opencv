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

#include <lib/contours.h>


///////////////////////////////////////////////////////////////////
//////////////////////// Lines Utils //////////////////////////////
///////////////////////////////////////////////////////////////////

cv::Point LineCoordinatesAddLength(const cv::Point &p1, const cv::Point &p2, const double &length, const double &addedLength) // get new point coordinates on a line from 1st point based on length
    // p1 and p2 define the line
    // length is the length from p1, direction p1 -> p2
    // addedLength is what is added to the length to form a new segment (it can be negative !), result is the 3rd point defining this new segment
{
    double x = ((addedLength * p1.x) + (length * p2.x)) / (length + addedLength);
    double y = ((addedLength * p1.y) + (length * p2.y)) / (length + addedLength);

    return cv::Point(x, y);
}

int LineCoefficients(const cv::Point &p1, const cv::Point &p2, double &m, double &p) // from two points get line equation using slope formula
    // line equation : y = mx + p --- which can also be x = (y - p) / m
    //      where m = (yB - yA) / (xB - xA)
    //            p = yA - m * xA
{
    if (p1.x == p2.x) { // line is vertical -> coefficients undefined
        m = std::numeric_limits<double>::quiet_NaN();
        p = p1.x;
        return line_vertical;
    }
    if (p1.y == p2.y) { // line is horizontal -> only p matters
        m = 0;
        p = p1.y;
        return line_horizontal;
    }

    m = double(p2.y - p1.y) / double(p2.x - p1.x);
    p = p1.y - m * p1.x;

    return line_oblique;
}

double LineCoordinateFromX(const double &m, const double &p, const int &lineType, const double &x) // get line coordinate Y from X using slope formula
    // line equation : y = mx + p
{
    if (lineType == line_oblique)
        return m * x + p;

    if (lineType == line_horizontal)
        return p;

    return x;
}

double LineCoordinateFromY(const double &m, const double &p, const int &lineType, const double &y) // get line coordinate X from Y using slope formula
    // line equation : x = (y - p) / m
{
    if (lineType == line_oblique)
        return (y - p) / m;

    if (lineType == line_vertical)
        return p;

    return y;
}

///////////////////////////////////////////////////////////////////
/////////////////////// Contours Utils ////////////////////////////
///////////////////////////////////////////////////////////////////

cv::Rect ContoursBoundingRect(const std::vector<std::vector<cv::Point>> &contours) // find bounding rectangle of array of contours
{
    int minX, minY, maxX, maxY; // to find limits
    minX = contours[0][0].x;
    minY = contours[0][0].y;
    maxX = contours[0][0].x;
    maxY = contours[0][0].y;

    for (int nb = 0; nb < int(contours.size()); nb++) { // parse contours
        cv::Rect rect = cv::boundingRect(contours[nb]); // rect for this contour
        if (rect.x < minX) // update limits if needed
            minX = rect.x;
        if (rect.y < minY)
            minY = rect.y;
        if (rect.x + rect.width > maxX)
            maxX = rect.x + rect.width;
        if (rect.y + rect.height > maxY)
            maxY = rect.y + rect.height;
    }

    return cv::Rect(minX, minY, maxX - minX, maxY - minY); // return rect with limits size
}

std::vector<cv::Point> SimpleContourFloatToInt(const std::vector<cv::Point2f> contour) // convert simple floats contour to integer
{
    std::vector<cv::Point> contourCopied;
    contourCopied.resize(contour.size());

    for (int nb = 0; nb < int(contour.size()); nb++)
        contourCopied[nb] = cv::Point(round(contour[nb].x), round(contour[nb].y));

    return contourCopied;
}

std::vector<cv::Point> CopySimpleContour(const std::vector<cv::Point> contour) // copy simple contour
{
    std::vector<cv::Point> contourCopied;
    contourCopied.resize(contour.size());

    for (int nb = 0; nb < int(contour.size()); nb++)
        contourCopied[nb] = contour[nb];

    return contourCopied;
}

std::vector<std::vector<cv::Point>> CopyContours(const std::vector<std::vector<cv::Point>> contours) // copy contours
{
    std::vector<std::vector<cv::Point>> contoursCopied;

    for (int nb = 0; nb < int(contours.size()); nb++) {
        contoursCopied.push_back(CopySimpleContour(contours[nb]));
    }

    return contoursCopied;
}

std::vector<std::vector<cv::Point>> ConvertSimpleContour(const std::vector<cv::Point> &contour) // some opencv functions need vector of vector
{
    std::vector<std::vector<cv::Point>> ret; // just copy contour to array of contours only containing this one
    ret.push_back(contour);

    return ret;
}

std::vector<std::vector<cv::Point>> ConvertSimpleContour2f(const std::vector<cv::Point2f> &contour) // some opencv functions need vector of vector
{
    std::vector<std::vector<cv::Point>> ret; // just copy contour to array of contours only containing this one
    std::vector<cv::Point> retTemp;
    for (int n = 0; n < int(contour.size()); n++)
        retTemp.push_back(cv::Point(round(contour[n].x), round(contour[n].y)));

    ret.push_back(retTemp);

    return ret;
}

///////////////////////////////////////////////////////////////////
////////////////////// Contours drawing ///////////////////////////
///////////////////////////////////////////////////////////////////

void DrawSimpleContour(cv::Mat &source, const std::vector<cv::Point> &contour,
                       const cv::Scalar &color, const int &thickness, const int &lineType,
                       const cv::Point &offset) // opencv drawcontours needs an array of array of points
{
    if (thickness == cv::FILLED) {
        cv::fillPoly(source, ConvertSimpleContour(contour), color, lineType, 0, offset);
        //cv::drawContours(source, ConvertSimpleContour(contour), 0, color, thickness, lineType, cv::noArray(), INT_MAX, offset); // the trick is to convert a simple contour to contours
    }
    else
        //cv::polylines(source, ConvertSimpleContour(contour), true, color, thickness, lineType, 0);
        cv::drawContours(source, ConvertSimpleContour(contour), 0, color, thickness, lineType, cv::noArray(), INT_MAX, offset); // the trick is to convert a simple contour to contours
}

void DrawContours(cv::Mat &source, const std::vector<std::vector<cv::Point>> &contours,
                  const cv::Scalar &color, const int &thickness, const int &lineType,
                  const cv::Point &offset) // draw several contours using DrawSimpleContour
{
    for (int nb = 0; nb < int(contours.size()); nb++) {
        DrawSimpleContour(source, contours[nb], color, thickness, lineType, offset);
    }
}

void ContoursToMask(const std::vector<std::vector<cv::Point>> &contours, const std::vector<cv::Vec4i> &hierarchy, const int &index,
                    cv::Mat &mask, cv::Rect & rect) // return a mask + bounding rectangle from contours (+ hierarchy)
{
    rect = cv::boundingRect(contours[index]); // find bounding rectangle for this contour
    mask = cv::Mat::zeros(rect.height, rect.width, CV_8UC1); // mask for this contour
    cv::drawContours(mask, contours, index, 1, cv::FILLED, cv::LINE_8, hierarchy, INT_MAX, cv::Point(-rect.x, -rect.y)); // draw filled poly from contour in mask
}

void SimpleContourToMask(const std::vector<cv::Point> &contour, cv::Mat &mask, cv::Rect & rect) // return a mask + bounding rectangle from simple contour
{
    rect = cv::boundingRect(contour); // find bounding rectangle for this contour
    mask = cv::Mat::zeros(rect.height, rect.width, CV_8UC1); // mask for this contour
    cv::drawContours(mask, ConvertSimpleContour(contour), 0, 1, cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, cv::Point(-rect.x, -rect.y)); // draw filled poly from contour in mask
}

cv::Mat SimpleContoursIntersectionToMask(const std::vector<cv::Point> &contour1, const std::vector<cv::Point> &contour2, int &area) // return a mask of 2 contours intersection
{
    cv::Rect rect1 = cv::boundingRect(contour1);
    cv::Rect rect2 = cv::boundingRect(contour2);

    cv::Point min = cv::Point(std::min(rect1.x, rect2.x), std::min(rect1.y, rect2.y));
    cv::Point max = cv::Point(std::max(rect1.x + rect1.width, rect2.x + rect2.width), std::max(rect1.y + rect1.height, rect2.y + rect2.height));

    cv::Mat mask1 = cv::Mat::ones(max.y - min.y, max.x - min.x, CV_8UC1);
    cv::drawContours(mask1, ConvertSimpleContour(contour1), 0, 0, cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, -min); // draw filled poly from contour in mask
    cv::Mat mask2 = cv::Mat::ones(max.y - min.y, max.x - min.x, CV_8UC1);
    cv::drawContours(mask2, ConvertSimpleContour(contour2), 0, 0, cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, -min); // draw filled poly from contour in mask

    cv::Mat result;
    cv::bitwise_or(mask1, mask2, result);

    result = 1 - result;

    area = cv::countNonZero(result);

    return result;
}

cv::Mat SimpleContoursIntersectionToMask2f(const std::vector<cv::Point2f> &contour1, const std::vector<cv::Point2f> &contour2, int &area) // return a mask of 2 contours intersection
{
    cv::Rect rect1 = cv::boundingRect(contour1);
    cv::Rect rect2 = cv::boundingRect(contour2);

    cv::Point2f min = cv::Point2f(std::min(rect1.x, rect2.x), std::min(rect1.y, rect2.y));
    cv::Point2f max = cv::Point2f(std::max(rect1.x + rect1.width, rect2.x + rect2.width), std::max(rect1.y + rect1.height, rect2.y + rect2.height));

    cv::Mat mask1 = cv::Mat::ones(max.y - min.y, max.x - min.x, CV_8UC1);
    cv::drawContours(mask1, ConvertSimpleContour2f(contour1), 0, 0, cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, -min); // draw filled poly from contour in mask
    cv::Mat mask2 = cv::Mat::ones(max.y - min.y, max.x - min.x, CV_8UC1);
    cv::drawContours(mask2, ConvertSimpleContour2f(contour2), 0, 0, cv::FILLED, cv::LINE_8, cv::noArray(), INT_MAX, -min); // draw filled poly from contour in mask

    cv::Mat result;
    cv::bitwise_or(mask1, mask2, result);
    result = 1 - result;

    area = cv::countNonZero(result);

    return result;
}

///////////////////////////////////////////////////////////////////
/////////////////////////// Transforms ////////////////////////////
///////////////////////////////////////////////////////////////////

void ScaleSimpleContour(std::vector<cv::Point> &contour, const float &scaleX, const float &scaleY, const int &width, const int &height) // rescale a contour
    // contour is scaled around the center of its bounding rectangle
    // either indicate a scale in percents, a width or height in pixels
    // if one of the two values (scale or length) is 0, the other is set to the same value
{
    if ((scaleX == 1.0) and ((scaleY == 0) or (scaleY == 1.0)))
        return;

    cv::Rect rct = cv::boundingRect(contour); // bounding rectangle of contour

    float ratioX = 0;
    float ratioY = 0;

    if (abs(scaleX) > 0) {
        ratioX = scaleX;
        if (scaleY == 0)
            ratioY = ratioX;
    }
    else if (abs(width) > 0) {
        ratioX = double(width) / double(rct.width);
        if (height == 0)
            ratioY = ratioX;
    }

    if (abs(scaleY) > 0) {
        ratioY = scaleY;
        if (scaleX == 0)
            ratioX = ratioY;
    }
    else if (abs(height) > 0) {
        ratioY = double(height) / double(rct.height);
        if (width == 0)
            ratioX = ratioY;
    }

    if (((ratioX == 0) or (ratioY == 0)) or ((ratioX == 1.0) and (ratioY == 1.0)))
        return;

    cv::Point rct_offset(-rct.tl().x, -rct.tl().y); // top-left of rectangle
    TranslateSimpleContour(contour, rct_offset.x, rct_offset.y); // shift contour by offset

    // multiply contour by ratio
    for (int i = 0; i < int(contour.size()); i++)
        contour[i] = cv::Point(contour[i].x * ratioX, contour[i].y * ratioY);

    cv::Rect rct_scale = cv::boundingRect(contour); // new bounding rectangle of contour

    // shift contour again
    cv::Point offset((rct.width - rct_scale.width) / 2, (rct.height - rct_scale.height) / 2); // new center of contour
    offset -= rct_offset; // take in account old center
    TranslateSimpleContour(contour, offset.x, offset.y); // shift contour with center
}

void MultiplyContours(std::vector<std::vector<cv::Point>> &contours, const float &scale) // multiply array of contours by a scale
    // coordinates are just multiplied by a scale
{
    if (scale == 1.0)
        return;

    // multiply contour by ratio
    for (int nb = 0; nb < int(contours.size()); nb++)
        for (int i = 0; i < int(contours[nb].size()); i++)
            contours[nb][i] = cv::Point(contours[nb][i].x * scale, contours[nb][i].y * scale);
}

void ScaleContours(std::vector<std::vector<cv::Point>> &contours, const float &scaleX, const float &scaleY, const int &width, const int &height) // rescale array of contours
    // contours are scaled around the center of their global bounding rectangle
    // either indicate a scale in percents, a width or height in pixels
    // if one of the two values (scale or length) is 0, the other is set to the same value
{
    if ((scaleX == 1.0) and ((scaleY == 0) or (scaleY == 1.0)))
        return;

    cv::Rect rct = ContoursBoundingRect(contours); // bounding rectangle of contour

    float ratioX = 0;
    float ratioY = 0;

    if (abs(scaleX) > 0) {
        ratioX = scaleX;
        if (scaleY == 0)
            ratioY = ratioX;
    }
    else if (abs(width) > 0) {
        ratioX = double(width) / double(rct.width);
        if (height == 0)
            ratioY = ratioX;
    }

    if (abs(scaleY) > 0) {
        ratioY = scaleY;
        if (scaleX == 0)
            ratioX = ratioY;
    }
    else if (abs(height) > 0) {
        ratioY = double(height) / double(rct.height);
        if (width == 0)
            ratioX = ratioY;
    }

    if (((ratioX == 0) or (ratioY == 0)) or ((ratioX == 1.0) and (ratioY == 1.0)))
        return;

    cv::Point rct_offset(-rct.tl().x, -rct.tl().y); // top-left of rectangle
    for (int nb = 0; nb < int(contours.size()); nb++)
        TranslateSimpleContour(contours[nb], rct_offset.x, rct_offset.y); // shift contour by offset

    // multiply contour by ratio
    for (int nb = 0; nb < int(contours.size()); nb++)
        for (int i = 0; i < int(contours[nb].size()); i++)
            contours[nb][i] = cv::Point(contours[nb][i].x * ratioX, contours[nb][i].y * ratioY);

    cv::Rect rct_scale = ContoursBoundingRect(contours); // new bounding rectangle of contour

    // shift contours again
    cv::Point offset((rct.width - rct_scale.width) / 2, (rct.height - rct_scale.height) / 2); // new center of contour
    offset -= rct_offset; // take in account old center
    for (int nb = 0; nb < int(contours.size()); nb++)
        TranslateSimpleContour(contours[nb], offset.x, offset.y); // shift contour with center
}

void MirrorSimpleContour(std::vector<cv::Point> &contour, const bool &horizontal, const bool &vertical) // mirror simple contour along vertical and/or horizontal axis
    // contour is mirrored around the center of its bounding rectangle
{
    cv::Rect2f rct = cv::boundingRect(contour); // bounding rectangle of contour
    cv::Point2f center = cv::Point2f(rct.x + rct.width / 2.0, rct.y + rct.height / 2.0);

    for (int nb = 0; nb < int(contour.size()); nb++) {
        if (horizontal)
            contour[nb].y = 2 * center.y - contour[nb].y;
        if (vertical)
            contour[nb].x = 2 * center.x - contour[nb].x;
    }
}

void MirrorContours(std::vector<std::vector<cv::Point>> &contours, const bool &horizontal, const bool &vertical) // mirror array of contours along vertical and/or horizontal axis
    // contour is mirrored around the center of its bounding rectangle
{
    cv::Rect rct = ContoursBoundingRect(contours); // bounding rectangle of contours
    cv::Point2f center = cv::Point2f(rct.x + rct.width / 2.0, rct.y + rct.height / 2.0);

    for (int nb = 0; nb < int(contours.size()); nb++) {
        for (int i = 0; i < int(contours[nb].size()); i++) {
            if (horizontal)
                contours[nb][i].y = 2 * center.y - contours[nb][i].y;
            if (vertical)
                contours[nb][i].x = 2 * center.x - contours[nb][i].x;
        }
    }
}

void RotateSimpleContour(std::vector<cv::Point> &contour, const float &angleRad, const cv::Point2f &center) // rotate a contour around a center
    // if center is not provided, rotation will use the center of the bounding rectangle of contour
{
    cv::Point2f contourCenter;
    if (center == cv::Point2f()) {
        cv::Rect2f rct = cv::boundingRect(contour); // bounding rectangle of contour
        contourCenter = cv::Point2f(rct.x + rct.width / 2.0, rct.y + rct.height / 2.0);
    }
    else {
        contourCenter = center;
    }

    for (int nb = 0; nb < int(contour.size()); nb++) {
        contour[nb] = cv::Point(cos(angleRad) * (contour[nb].x - contourCenter.x) - sin(angleRad) * (contour[nb].y - contourCenter.y) + contourCenter.x,
                                sin(angleRad) * (contour[nb].x - contourCenter.x) + cos(angleRad) * (contour[nb].y - contourCenter.y) + contourCenter.y);
    }
}

void RotateContours(std::vector<std::vector<cv::Point>> &contours, const float &angleRad, const cv::Point2f &center) // rotate array of contours around a center
    // if center is not provided, rotation will use the center of the global bounding rectangle of contours
{
    cv::Point2f contoursCenter;
    if (center == cv::Point2f()) {
        cv::Rect2f rct = ContoursBoundingRect(contours); // bounding rectangle of contour
        contoursCenter = cv::Point2f(rct.x + rct.width / 2.0, rct.y + rct.height / 2.0);
    }
    else {
        contoursCenter = center;
    }

    for (int nb = 0; nb < int(contours.size()); nb++)
        RotateSimpleContour(contours[nb], angleRad, contoursCenter);
}

void WarpSimpleContourHorizontal(std::vector<cv::Point> &contour, const int &typeH, const int &pixelsX) // warp a simple contour (horizontal)
{
    cv::Rect rct = cv::boundingRect(contour); // bounding rectangle of contour

    for (int nb = 0; nb < int(contour.size()); nb++)
        contour[nb].x += WarpCurve(typeH, contour[nb].y - rct.y, rct.width, rct.height, pixelsX);
}

void WarpSimpleContourVertical(std::vector<cv::Point> &contour, const int &typeH, const int &pixelsY) // warp a simple contour (vertical)
{
    cv::Rect rct = cv::boundingRect(contour); // bounding rectangle of contour

    for (int nb = 0; nb < int(contour.size()); nb++)
        contour[nb].y += WarpCurve(typeH, contour[nb].x - rct.x, rct.width, rct.height, pixelsY);
}

void WarpContoursHorizontal(std::vector<std::vector<cv::Point>> &contours, const int &typeH, const int &pixelsX) // warp array of contours (horizontal)
{
    cv::Rect rct = ContoursBoundingRect(contours); // bounding rectangle of contour

    for (int nb = 0; nb < int(contours.size()); nb++)
        for (int i = 0; i < int(contours[nb].size()); i++)
            contours[nb][i].x += WarpCurve(typeH, contours[nb][i].y - rct.y, rct.width, rct.height, pixelsX);
}

void WarpContoursVertical(std::vector<std::vector<cv::Point>> &contours, const int &typeH, const int &pixelsY) // warp array of contours (vertical)
{
    cv::Rect rct = ContoursBoundingRect(contours); // bounding rectangle of contour

    for (int nb = 0; nb < int(contours.size()); nb++)
        for (int i = 0; i < int(contours[nb].size()); i++)
            contours[nb][i].y += WarpCurve(typeH, contours[nb][i].x - rct.x, rct.width, rct.height, pixelsY);
}

void TranslateSimpleContour(std::vector<cv::Point> &contour, const int &shiftHorizontal, const int &shiftVertical) // translate simple contour along vertical and/or horizontal axis
{
    cv::Point t = cv::Point(shiftHorizontal, shiftVertical);
    for (int nb = 0; nb < int(contour.size()); nb++)
        contour[nb] += t;
}

void TranslateContours(std::vector<std::vector<cv::Point>> &contours, const int &shiftHorizontal, const int &shiftVertical) // translate contours along vertical and/or horizontal axis
{
    for (int nb = 0; nb < int(contours.size()); nb++)
        TranslateSimpleContour(contours[nb], shiftHorizontal, shiftVertical);
}

///////////////////////////////////////////////////////////////////
///////////////////////// Interpolation ///////////////////////////
///////////////////////////////////////////////////////////////////

/*
 * Compute coefficients for a cubic polynomial
 *   p(s) = c0 + c1*s + c2*s^2 + c3*s^3
 * such that
 *   p(0) = x0, p(1) = x1
 *  and
 *   p'(0) = t0, p'(1) = t1
 */
void InitCubicPoly(float x0, float x1, float t0, float t1, CubicPoly &p)
{
    p.c0 = x0;
    p.c1 = t0;
    p.c2 = -3 * x0 + 3 * x1 - 2 * t0 - t1;
    p.c3 =  2 * x0 - 2 * x1 + t0 + t1;
}

/*void InitCatmullRom(float x0, float x1, float x2, float x3, CubicPoly &p) // standard Catmull-Rom spline: interpolate between x1 and x2 with previous/following points x1/x4 - not used here, just for illustration
{
    // Catmull-Rom with tension 0.5
    InitCubicPoly(x1, x2, 0.5f * (x2 - x0), 0.5f * (x3 - x1), p);
}*/

void InitNonUniformCatmullRom(float x0, float x1, float x2, float x3, float dt0, float dt1, float dt2, CubicPoly &p) // compute coefficients for a non-uniform Catmull-Rom spline
{
    // compute tangents
    float t1 = (x1 - x0) / dt0 - (x2 - x0) / (dt0 + dt1) + (x2 - x1) / dt1;
    float t2 = (x2 - x1) / dt1 - (x3 - x1) / (dt1 + dt2) + (x3 - x2) / dt2;

    // rescale tangents for parametrization in [0,1]
    t1 *= dt1;
    t2 *= dt1;

    InitCubicPoly(x1, x2, t1, t2, p);
}

float VecDistSquared(const cv::Point2f &p, const cv::Point2f &q) // quick distance, squared
{
    return (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y);
}

void InitCatmullRom(const cv::Point2f &p0, const cv::Point2f &p1, const cv::Point2f &p2, const cv::Point2f &p3, const splineType &sType, CubicPoly &px, CubicPoly &py)
{
    float alpha;
    switch (sType) {
        case spline_centripetal: alpha = 0.25f; break; // 0.5 = centripetal
        case spline_cordal:      alpha = 0.0f;  break; // 0 = cordal
        case spline_uniform:     alpha = 0.5f;  break; // 1.0 = uniform
    }

    float dt0 = pow(VecDistSquared(p0, p1), alpha);
    float dt1 = pow(VecDistSquared(p1, p2), alpha);
    float dt2 = pow(VecDistSquared(p2, p3), alpha);

    // safety check for repeated points
    if (dt1 < 1e-4f)
        dt1 = 1.0f;
    if (dt0 < 1e-4f)
        dt0 = dt1;
    if (dt2 < 1e-4f)
        dt2 = dt1;

    InitNonUniformCatmullRom(p0.x, p1.x, p2.x, p3.x, dt0, dt1, dt2, px);
    InitNonUniformCatmullRom(p0.y, p1.y, p2.y, p3.y, dt0, dt1, dt2, py);
}

std::vector<cv::Point> GetCurvedLineCatmullRom(const cv::Point &p0, const cv::Point &p1, const cv::Point &p2, const cv::Point &p3, const splineType &sType, const int &nbSegments) // interpolate a line from 4 points using non-uniform Catmull-Rom formula
{
    CubicPoly px, py;
    std::vector<cv::Point> result;
    float invSegments = 1.0f / nbSegments;

    InitCatmullRom(p0, p1, p2, p3, sType, px, py);
    for (int i = 0; i <= nbSegments; ++i) {
        result.push_back(cv::Point(px.eval(invSegments * i), py.eval(invSegments * i)));
    }

    return result;
}
