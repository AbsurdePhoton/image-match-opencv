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


#include "lib/image-utils.h"
#include <fstream>


///////////////////////////////////////////////////////////
//// General
///////////////////////////////////////////////////////////

int GetByteInRange(const int &byte) // get a byte in range [0..255]
{
    if (byte < 0)
        return 0;
    else if (byte > 255)
        return 255;

    // if other conditions fail
    return byte;
}

///////////////////////////////////////////////////////////
//// Utils
///////////////////////////////////////////////////////////

int CountRGBUniqueValues(const cv::Mat &source) // count number of RGB colors in BGR non-alpha image
{
    std::set<int> unique;

    for (cv::Vec3b &p : cv::Mat_<cv::Vec3b>(source)) // iterate over pixels (assummes CV_8UC3 !)
        unique.insert((p[2] << 16) | (p[1] << 8) | (p[0])); // "hash" representation of the pixel

    return unique.size();
}

cv::Mat NormalizeImage(const cv::Mat & source, const bool &multiThread) // normalize [0..255] image to [0..1] - returns a CV_64F image
{
    cv::Mat result;

    switch (source.channels()) {
        case 1:
            result = cv::Mat(source.rows, source.cols, CV_64FC1); break;
        case 2:
            result = cv::Mat(source.rows, source.cols, CV_64FC2); break;
        case 3:
            result = cv::Mat(source.rows, source.cols, CV_64FC3); break;
        case 4:
            result = cv::Mat(source.rows, source.cols, CV_64FC4);
    }

    const uchar* sourceP = source.ptr<uchar>(0);
    double* resultP = result.ptr<double>(0);

    int total = source.cols * source.rows * source.channels();
    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int n = 0; n < total; n++)
            resultP[n] = double(sourceP[n]) / 255.0;
    }

    return result;
}

cv::Mat DeNormalizeImage(const cv::Mat & source, const bool &multiThread) // denormalize [0..1] image to [0..255] - needs a CV_64F input
{
    cv::Mat result;

    switch (source.channels()) {
        case 1:
            result = cv::Mat(source.rows, source.cols, CV_8UC1); break;
        case 2:
            result = cv::Mat(source.rows, source.cols, CV_8UC2); break;
        case 3:
            result = cv::Mat(source.rows, source.cols, CV_8UC3); break;
        case 4:
            result = cv::Mat(source.rows, source.cols, CV_8UC4); break;
    }

    const double* sourceP = source.ptr<double>(0);
    uchar* resultP = result.ptr<uchar>(0);

    int total = source.cols * source.rows * source.channels();
    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int n = 0; n < total; n++)
            resultP[n] = std::round(sourceP[n] * 255.0);
    }

    return result;
}

bool MatEqual(const cv::Mat &one, const cv::Mat &two) // compare two Mat
{
    // one image empty ?
    if ((one.empty()) or (two.empty()))
        return false;

    // image dimensions and channels different ?
    if ((one.channels() != two.channels()) or (one.cols != two.cols) or (one.rows != two.rows))
        return false;

    cv::Mat dif;
    cv::compare(one, two, dif, cv::CMP_NE); // compare the 2 images
    int nz = cv::countNonZero(dif); // if there is no difference, there are only zero values

    return nz == 0; // if there are only zero values, the result is true (the 2 images are equal)
}

bool IsImageSolidColor(const cv::Mat &source) // tells if BGR image consists of a unique color
{
   cv::Vec3b value = source.at<cv::Vec3b>(0, 0);
   const cv::Vec3b *sourceP = source.ptr<cv::Vec3b>(0);
   bool same = true;
   int total = source.cols * source.rows;
   for (int n = 0; n < total; n++) {
       if (sourceP[n] != value) {
           same = false;
           break;
       }
   }

   return same;
}

void ClampMat(cv::Mat &source, const double &min, const double &max) // clamp matrix values between min and max
{
    cv::min(source, max, source);
    cv::max(source, min, source);
}

void SaveFloatMatToFile(const cv::Mat &m, const char* filename) // save a float cv::Mat to ascii file
{
    std::ofstream fOut(filename);

    for (int i = 0; i < m.rows; i++) {
        for(int j=0; j<m.cols; j++) {
            fOut << m.at<float>(i, j) << "\t";
        }
        fOut << std::endl;
    }

    fOut.close();
}

cv::Mat ImageAnydepthToColor(const cv::Mat &source) // convert any image of any depth (even grayscale) to BGR
    // for example for a cv::Mat loaded with cv::imread with the option cv::IMREAD_ANYDEPTH
    // output is BGR even for grayscale images
    // transparency is taken in account and show a grid
{
    cv::Mat img = source.clone();

    if (!img.empty()) {
        // to 8-bit integers
        if (img.depth() != CV_8U) {
            double minVal, maxVal;
            cv::minMaxLoc(img, &minVal, &maxVal);
            if ((maxVal <= 1.0) or (maxVal > 255.0))
                cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);

            img.convertTo(img, CV_8U);
        }

        // alpha grayscale case
        if (img.channels() == 2) {
            std::vector<cv::Mat> tmp;
            cv::split(img, tmp);
            cv::cvtColor(tmp[0], img, cv::COLOR_GRAY2BGR);
            img = AddAlphaToImage(img, tmp[1]);
        }

        // grayscale to "color"
        if (img.channels() == 1)
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

        // color + alpha
        if (img.channels() == 4)
            img = ImageAlphaWithGrid(img);

        return img; // return converted image from OpenCV to Qt format
    }

    return cv::Mat();
}

///////////////////////////////////////////////////////////
//// Conversions from QImage and QPixmap to Mat
///////////////////////////////////////////////////////////

cv::Mat QImage2Mat(const QImage &source) // Convert QImage to Mat, every number of channels supported
{
    switch (source.format()) { // which forcv::Mat ?

        case QImage::Format_ARGB32: // 8-bit, 4 channel
        case QImage::Format_ARGB32_Premultiplied: {
            cv::Mat temp(source.height(), source.width(), CV_8UC4,
                        const_cast<uchar*>(source.bits()),
                        static_cast<size_t>(source.bytesPerLine()));
            return temp;
        }

        case QImage::Format_RGB32: { // 8-bit, 3 channel
            cv::Mat temp(source.height(), source.width(), CV_8UC4,
                        const_cast<uchar*>(source.bits()),
                        static_cast<size_t>(source.bytesPerLine()));
            cv::Mat tempNoAlpha;
            cv::cvtColor(temp, tempNoAlpha, cv::COLOR_BGRA2BGR);   // drop the all-white alpha channel
        return tempNoAlpha;
        }

        case QImage::Format_RGB888: { // 8-bit, 3 channel
            QImage swapped = source.rgbSwapped();
            return cv::Mat(swapped.height(), swapped.width(), CV_8UC3,
                       const_cast<uchar*>(swapped.bits()),
                       static_cast<size_t>(swapped.bytesPerLine())).clone();
        }

        case QImage::Format_Indexed8: { // 8-bit, 1 channel
            cv::Mat temp(source.height(), source.width(), CV_8UC1,
                        const_cast<uchar*>(source.bits()),
                        static_cast<size_t>(source.bytesPerLine()));
            return temp;
        }

        default:
            break;
    }

    return cv::Mat(); // return empty cv::Mat if type not found
}

cv::Mat QPixmap2Mat(const QPixmap &source) // Convert Pixmap to Mat
{
    return QImage2Mat(source.toImage()); // simple !
}

///////////////////////////////////////////////////////////
//// cv::Mat conversions to QImage and QPixmap
///////////////////////////////////////////////////////////

QImage Mat2QImage(const cv::Mat &source) // convert BGR cv::Mat to RGB QImage
{
    if (source.empty())
        return QImage();

     cv::Mat temp;
     cv::cvtColor(source, temp, cv::COLOR_BGR2RGB); // convert cv::Mat BGR to QImage RGB
     QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888); // conversion
     dest.bits(); // enforce deep copy of QImage::QImage (const uchar * data, int width, int height, Forcv::Mat format)

     return dest;
}

QPixmap Mat2QPixmap(const cv::Mat &source) // convert cv::Mat to QPixmap
{
    if (source.empty())
        return QPixmap();

    QImage i = Mat2QImage(source); // first step: convert to QImage
    QPixmap p = QPixmap::fromImage(i, Qt::AutoColor); // then convert QImage to QPixmap

    return p;
}

QPixmap Mat2QPixmapResized(const cv::Mat &source, const int &width, const int &height, const bool &smooth) // convert cv::Mat to resized QPixmap
{
    if (source.empty())
        return QPixmap();

    Qt::TransformationMode quality; // quality
    if (smooth) quality = Qt::SmoothTransformation;
        else quality = Qt::FastTransformation;

    QImage i = Mat2QImage(source); // first step: convert to QImage
    QPixmap p = QPixmap::fromImage(i, Qt::AutoColor); // then convert QImage to QPixmap

    return p.scaled(width, height, Qt::IgnoreAspectRatio, quality); // resize with high quality
}

QImage cvMatToQImage(const cv::Mat &source) // another implementation BGR to RGB, every number of channels supported
{
    if (source.empty())
        return QImage();

    switch (source.type()) { // which type ?

        case CV_8UC4: { // 8-bit, 4 channel
            QImage image(   source.data,
                            source.cols, source.rows,
                            static_cast<int>(source.step),
                            QImage::Format_ARGB32);
            image.bits(); // enforce deep copy of QImage::QImage (const uchar * data, int width, int height, Forcv::Mat format)
            return image;
        }


        case CV_8UC3: { // 8-bit, 3 channel
            QImage image(   source.data,
                            source.cols, source.rows,
                            static_cast<int>(source.step),
                            QImage::Format_RGB888);
            image.bits(); // enforce deep copy of QImage::QImage (const uchar * data, int width, int height, Forcv::Mat format)
            return image.rgbSwapped();
        }

        case CV_8UC1: { // 8-bit, 1 channel
            QImage image(source.data, source.cols, source.rows,
                         static_cast<int>(source.step),
                         QImage::Format_Grayscale8);
            image.bits(); // enforce deep copy of QImage::QImage (const uchar * data, int width, int height, Forcv::Mat format)
            return image;
        }
    }

    return QImage(); // return empty cv::Mat if type not found
}

///////////////////////////////////////////////////////////
//// Copy and Paste
///////////////////////////////////////////////////////////

cv::Rect RectangleFromImage(const cv::Mat &source, const cv::Rect &frame) // get a rectangle cropped to the image limits
{
    int minX = std::max(0, frame.x);
    int minY = std::max(0, frame.y);
    int maxX = std::min(frame.x + frame.width, source.cols);
    int maxY = std::min(frame.y + frame.height, source.rows);

    cv::Rect sourceRect = cv::Rect(minX, minY, maxX - minX, maxY - minY); // source rect to copy from

    return sourceRect;
}

cv::Mat CopyFromImage(const cv::Mat &source, const cv::Rect &frame) // copy part of an image - if parts of frame outside source image, non-copied pixels are black
{
    cv::Mat dest = cv::Mat::zeros(frame.height, frame.width, source.type()); // result is the same size as frame

    cv::Rect sourceRect = RectangleFromImage(source, frame);
    if ((sourceRect.width <= 0) or (sourceRect.height <= 0))
        return dest;

    cv::Rect destRect = sourceRect - frame.tl(); // destination rect to copy to

    source(sourceRect).copyTo(dest(destRect)); // copy (eventually resized) frame to destination

    return dest;
}

void PasteImageGraySlow(cv::Mat &background, const cv::Mat &foreground,
                        const int &originX, const int &originY,
                        const bool &transparency, const uchar &transparentColor,
                        const bool &multiThread) // copy gray image onto another with optional transparency
    // for pasting without transparency, use instead PasteImageFast, it is way faster !
{
    // compute limits of image to paste
    int minX = std::max(0, -originX);
    int minY = std::max(0, -originY);
    int maxX = std::min(foreground.cols, background.cols - originX);
    int maxY = std::min(foreground.rows, background.rows - originY);

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar* backgroundP = background.ptr<uchar>(j + originY);
            const uchar* foregroundP = foreground.ptr<uchar>(j);

            for (int i = minX; i < maxX; i++) {
                if (!transparency) {
                    backgroundP[i + originX] = foregroundP[i];
                }
                else {
                    uchar pixel = foregroundP[i]; // current pixel
                    if (pixel != transparentColor) // is it transparent ?
                        backgroundP[i + originX] = pixel; // copy non-zero values only
                }
            }
        }
    }
}

void PasteImageColorSlow(cv::Mat &background, const cv::Mat &foreground,
                         const int &originX, const int &originY,
                         const bool &transparency, const cv::Vec3b &transparentColor,
                         const bool &multiThread) // copy BGR image onto another with optional transparency
    // for pasting without transparency, use instead PasteImageFast, it is way faster !
{
    // compute limits of image to paste
    int minX = std::max(0, -originX);
    int minY = std::max(0, -originY);
    int maxX = std::min(foreground.cols, background.cols - originX);
    int maxY = std::min(foreground.rows, background.rows - originY);

    cv::Vec3b pixel;

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            cv::Vec3b* backgroundP = background.ptr<cv::Vec3b>(j + originY);
            const cv::Vec3b* foregroundP = foreground.ptr<cv::Vec3b>(j);

            for (int i = minX; i < maxX; i++) {
                if (!transparency) {
                    backgroundP[i + originX] = foregroundP[i];
                }
                else {
                    pixel = foregroundP[i]; // current pixel
                    if (pixel != transparentColor) // is it transparent ?
                        backgroundP[i + originX] = pixel; // copy non-zero values only
                }
            }
        }
    }
}

void PasteImageGrayTransparency(cv::Mat &background, const cv::Mat &foreground,
                                const int &originX, const int &originY,
                                const uchar &transparentColor, const bool &multiThread) // copy gray image onto another with transparency
    // for pasting without transparency, use instead PasteImageFast, it is way faster !
{
    int minX = std::max(0, -originX);
    int minY = std::max(0, -originY);
    int maxX = std::min(foreground.cols, background.cols - originX);
    int maxY = std::min(foreground.rows, background.rows - originY);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    uchar *ptrB = background.ptr(0);
    const uchar *ptrF = foreground.ptr(0);

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar *PB = ptrB + (j + originY) * stepB + originX;
            const uchar *PF = ptrF + j * stepF;
            for (int i = minX; i < maxX; i++) {
                uchar pixel;
                std::memcpy(&pixel, PF + i, 1);
                if (pixel != transparentColor) // is it transparent ?
                    std::memcpy(PB + i, &pixel, 1);
            }
        }
    }
}

void PasteImageColorTransparency(cv::Mat &background, const cv::Mat &foreground,
                                 const int &originX, const int &originY,
                                 const cv::Vec3b &transparentColor, const bool &multiThread) // copy BGR image onto another with transparency
    // for pasting without transparency, use instead PasteImageFast, it is way faster !
{
    int minX = std::max(0, -originX);
    int minY = std::max(0, -originY);
    int maxX = std::min(foreground.cols, background.cols - originX);
    int maxY = std::min(foreground.rows, background.rows - originY);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    int pix_size = (int)background.elemSize();
    uchar *ptrB = background.ptr(0);
    const uchar *ptrF = foreground.ptr(0);

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar *PB = ptrB + (j + originY) * stepB + originX * pix_size;
            const uchar *PF = ptrF + j * stepF;
            for (int i = minX; i < maxX; i++) {
                cv::Vec3b pixel;
                std::memcpy(&pixel, PF + i * pix_size, pix_size);
                if (pixel != transparentColor) // is it transparent ?
                    std::memcpy(PB + i * pix_size, &pixel, pix_size);
            }
        }
    }
}

void PasteImageAlphaSlow(cv::Mat &background, const cv::Mat &foreground,
                         const cv::Point &pos, const bool &multiThread) // paste alpha BGR image on non-alpha background
    //  a faster version is PasteImageAlphaFast
{
    // compute limits of image to paste
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    for (int j = minY; j < maxY; j++) {
        const cv::Vec4b* foregroundP = foreground.ptr<cv::Vec4b>(j);
        cv::Vec3b* backgroundP = background.ptr<cv::Vec3b>(j + pos.y);
        #pragma omp parallel if(multiThread)
        {
            #pragma omp for nowait
            for (int i = minX; i < maxX; i++) {
                double alpha = double(foregroundP[i][3]) / 255.0;
                for (int c = 0; c < 3; c++) {
                    // formula : Intensity = alpha * Foreground + (1.0 - alpha) * Background (with alpha in [0..1])
                    backgroundP[i + pos.x][c] = GetByteInRange(alpha * double(foregroundP[i][c]) + (1.0 - alpha) * double(backgroundP[i + pos.x][c]));
                }
            }
        }
    }
}

void PasteImageGrayPlusAlphaOnColorSlow(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                                        const cv::Point pos, const bool &multiThread) // paste gray + alpha image on non-alpha BGR background
    // this function is not public, general case defined in PasteImagePlusAlpha
    // a faster version is defined with general case PasteImagePlusAlphaFast
{
    // compute limits of image to paste
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    for (int j = minY; j < maxY; j++) {
        const uchar* foregroundP = foreground.ptr<uchar>(j);
        const uchar* alphaP = alpha.ptr<uchar>(j);
        cv::Vec3b* backgroundP = background.ptr<cv::Vec3b>(j + pos.y);
        #pragma omp parallel if(multiThread)
        {
            #pragma omp for nowait
            for (int i = minX; i < maxX; i++) {
                double alphaValue = double(alphaP[i]) / 255.0;
                for (int c = 0; c < 3; c++) {
                    // formula : Intensity = alpha * Foreground + (1 - alpha) * Background (with alpha in [0..1])
                    backgroundP[i + pos.x][c] = GetByteInRange(alphaValue * double(foregroundP[i]) + (1.0 - alphaValue) * double(backgroundP[i + pos.x][c]));
                }
            }
        }
    }
}

void PasteImageColorPlusAlphaOnColorSlow(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                                         const cv::Point pos, const bool &multiThread) // paste BGR + alpha image on non-alpha BGR background
    // this function is not public, general case defined in PasteImagePlusAlpha
    // a faster version is defined with general case PasteImagePlusAlphaFast
{
    // compute limits of image to paste
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    for (int j = minY; j < maxY; j++) {
        const cv::Vec3b* foregroundP = foreground.ptr<cv::Vec3b>(j);
        const uchar* alphaP = alpha.ptr<uchar>(j);
        cv::Vec3b* backgroundP = background.ptr<cv::Vec3b>(j + pos.y);
        #pragma omp parallel if(multiThread)
        {
            #pragma omp for nowait
            for (int i = minX; i < maxX; i++) {
                double alphaValue = double(alphaP[i]) / 255.0;

                for (int c = 0; c < 3; c++) {
                    // formula : Intensity = alpha * Foreground + (1 - alpha) * Background (with alpha in [0..1])
                    backgroundP[i + pos.x][c] = GetByteInRange(alphaValue * double(foregroundP[i][c]) + (1.0 - alphaValue) * double(backgroundP[i + pos.x][c]));
                }
            }
        }
    }
}

void PasteImageGrayPlusAlphaOnGraySlow(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                                       const cv::Point pos, const bool &multiThread) // paste gray + alpha image on gray background
    // this function is not public, general case defined in PasteImagePlusAlpha
    // a faster version is defined with general case PasteImagePlusAlphaFast
{
    // compute limits of image to paste
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    for (int j = minY; j < maxY; j++) {
        const uchar* foregroundP = foreground.ptr<uchar>(j);
        const uchar* alphaP = alpha.ptr<uchar>(j);
        uchar* backgroundP = background.ptr<uchar>(j + pos.y);
        #pragma omp parallel if(multiThread)
        {
            #pragma omp for nowait
            for (int i = minX; i < maxX; i++) {
                double alphaValue = double(alphaP[i]) / 255.0;
                // formula : Intensity = alpha * Foreground + (1 - alpha) * Background (with alpha in [0..1])
                backgroundP[i + pos.x] = GetByteInRange(alphaValue * double(foregroundP[i]) + (1.0 - alphaValue) * double(backgroundP[i + pos.x]));
            }
        }
    }
}

void PasteImagePlusAlphaSlow(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                             const cv::Point &pos, const bool &multiThread) // paste (gray or color) + alpha image on non-alpha gray or BGR background
    // a faster version is defined with general case PasteImagePlusAlphaFast
{
    if (foreground.channels() == 1) { // image to paste if gray
        if (background.channels() == 1) // destination image is gray
            PasteImageGrayPlusAlphaOnGraySlow(background, foreground, alpha, pos, multiThread);
        else // destination image is BGR
            PasteImageGrayPlusAlphaOnColorSlow(background, foreground, alpha, pos, multiThread);
    }
    else if (foreground.channels() == 3) { // image to paste is BGR
        PasteImageColorPlusAlphaOnColorSlow(background, foreground, alpha, pos, multiThread);
    }
}

void PasteImageAlphaOnAlphaSlow(cv::Mat &background, const cv::Mat &foreground,
                                const cv::Point &pos, const bool &multiThread) // paste alpha BGR image on alpha BGR background
{
    // compute limits of image to paste
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    for (int j = minY; j < maxY; j++) {
        const cv::Vec4b* foregroundP = foreground.ptr<cv::Vec4b>(j);
        cv::Vec4b* backgroundP = background.ptr<cv::Vec4b>(j + pos.y);
        #pragma omp parallel if(multiThread)
        {
            #pragma omp for nowait
            for (int i = minX; i < maxX; i++) {
                double alpha = double(foregroundP[i][3]) / 255.0;
                for (int c = 0; c < 3; c++) {
                    // formula : Intensity = alpha * Foreground + (1 - alpha) * Background (with alpha in [0..1])
                    backgroundP[i + pos.x][c] = GetByteInRange(alpha * double(foregroundP[i][c]) + (1.0 - alpha) * double(backgroundP[i + pos.x][c]));
                }
                backgroundP[i + pos.x][3] = (backgroundP[i][3] / 255.0) * (foregroundP[i][3] / 255.0) * 255.0;
            }
        }
    }
}

void PasteImageFast(cv::Mat &background, const cv::Mat &foreground,
                    const int &originX, const int &originY) // copy a matrix onto another of any type - really fast implementation
{
    // compute limits of image to paste
    int minX = std::max(0, -originX);
    int minY = std::max(0, -originY);
    int maxX = std::min(foreground.cols, background.cols - originX);
    int maxY = std::min(foreground.rows, background.rows - originY);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    int pix_size = (int)background.elemSize();
    uchar *ptrB = background.ptr(0);
    const uchar *ptrF = foreground.ptr(0);
    int width = (maxX - minX) * pix_size;

    for (int j = minY; j < maxY; j++) {
        uchar *PB = ptrB + (j + originY) * stepB + (minX + originX) * pix_size;
        const uchar *PF = ptrF + j * stepF + minX * pix_size;
        memcpy(PB, PF, width);
    }
}

void PasteImageAlphaFast(cv::Mat &background, const cv::Mat &foreground,
                         const cv::Point &pos, const bool &multiThread) // paste alpha BGR image on non-alpha background - fast version
{
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    int pix_sizeB = (int)background.elemSize();
    int pix_sizeF = (int)foreground.elemSize();
    uchar *ptrB = background.ptr(0);
    const uchar *ptrF = foreground.ptr(0);

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar *PB = ptrB + (j + pos.y) * stepB + pos.x * pix_sizeB;
            const uchar *PF = ptrF + j * stepF;
            for (int i = minX; i < maxX; i++) {
                cv::Vec3b pixelB;
                std::memcpy(&pixelB, PB + i * pix_sizeB, pix_sizeB);
                cv::Vec4b pixelF;
                std::memcpy(&pixelF, PF + i * pix_sizeF, pix_sizeF);
                double alpha = double(pixelF[3]) / 255.0;
                for (int c = 0; c < 3; c++) {
                    // formula : Intensity = alpha * Foreground + (1.0 - alpha) * Background (with alpha in [0..1])
                    pixelB[c] = GetByteInRange(alpha * double(pixelF[c]) + (1.0 - alpha) * double(pixelB[c]));
                }
                std::memcpy(PB + i * pix_sizeB, &pixelB, pix_sizeB);
            }
        }
    }
}

void PasteImageAlphaOnAlphaFast(cv::Mat &background, const cv::Mat &foreground,
                                const cv::Point &pos, const bool &multiThread) // paste alpha BGR image on alpha BGR background - fast version
{
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    int pix_sizeB = (int)background.elemSize();
    int pix_sizeF = (int)foreground.elemSize();
    uchar *ptrB = background.ptr(0);
    const uchar *ptrF = foreground.ptr(0);

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar *PB = ptrB + (j + pos.y) * stepB + pos.x * pix_sizeB;
            const uchar *PF = ptrF + j * stepF;
            for (int i = minX; i < maxX; i++) {
                cv::Vec4b pixelB;
                std::memcpy(&pixelB, PB + i * pix_sizeB, pix_sizeB);
                cv::Vec4b pixelF;
                std::memcpy(&pixelF, PF + i * pix_sizeF, pix_sizeF);
                double alphaF = double(pixelF[3]) / 255.0;
                double alphaB = double(pixelB[3]) / 255.0;
                for (int c = 0; c < 3; c++) {
                    // formula : Intensity = alpha * Foreground + (1.0 - alpha) * Background (with alpha in [0..1])
                    pixelB[c] = GetByteInRange(alphaF * double(pixelF[c]) + (1.0 - alphaF) * double(pixelB[c]));
                }
                pixelB[3] = alphaF * alphaB * 255.0;
                std::memcpy(PB + i * pix_sizeB, &pixelB, pix_sizeB);
            }
        }
    }
}

void PasteImageGrayPlusAlphaOnColorFast(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                                        const cv::Point pos, const bool &multiThread) // paste gray + alpha image on non-alpha BGR background
{
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    size_t stepA = alpha.step;
    int pix_sizeB = (int)background.elemSize();
    int pix_sizeF = (int)foreground.elemSize();
    int pix_sizeA = (int)alpha.elemSize();
    uchar *ptrB = background.ptr(0);
    const uchar* ptrF = foreground.ptr(0);
    const uchar* ptrA = alpha.ptr(0);

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar *PB = ptrB + (j + pos.y) * stepB + pos.x * pix_sizeB;
            const uchar *PF = ptrF + j * stepF;
            const uchar *PA = ptrA + j * stepA;
            for (int i = minX; i < maxX; i++) {
                cv::Vec3b pixelB;
                std::memcpy(&pixelB, PB + i * pix_sizeB, pix_sizeB);
                uchar pixelF;
                std::memcpy(&pixelF, PF + i * pix_sizeF, pix_sizeF);
                uchar pixelA;
                std::memcpy(&pixelA, PA + i * pix_sizeA, pix_sizeA);
                double alphaValue = double(pixelA) / 255.0;
                for (int c = 0; c < 3; c++) {
                    // formula : Intensity = alpha * Foreground + (1.0 - alpha) * Background (with alpha in [0..1])
                    //backgroundP[i + pos.x][c] = GetByteInRange(alphaValue * double(foregroundP[i]) + (1.0 - alphaValue) * double(backgroundP[i + pos.x][c]));
                    pixelB[c] = GetByteInRange(alphaValue * double(pixelF) + (1.0 - alphaValue) * double(pixelB[c]));
                }
                std::memcpy(PB + i * pix_sizeB, &pixelB, pix_sizeB);
            }
        }
    }
}

void PasteImageColorPlusAlphaOnColorFast(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                                         const cv::Point pos, const bool &multiThread) // paste BGR + alpha image on non-alpha BGR background - fast version
{
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    size_t stepA = alpha.step;
    int pix_sizeB = (int)background.elemSize();
    int pix_sizeF = (int)foreground.elemSize();
    int pix_sizeA = (int)alpha.elemSize();
    uchar *ptrB = background.ptr(0);
    const uchar* ptrF = foreground.ptr(0);
    const uchar* ptrA = alpha.ptr(0);

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar *PB = ptrB + (j + pos.y) * stepB + pos.x * pix_sizeB;
            const uchar *PF = ptrF + j * stepF;
            const uchar *PA = ptrA + j * stepA;
            for (int i = minX; i < maxX; i++) {
                cv::Vec3b pixelB;
                std::memcpy(&pixelB, PB + i * pix_sizeB, pix_sizeB);
                cv::Vec3b pixelF;
                std::memcpy(&pixelF, PF + i * pix_sizeF, pix_sizeF);
                uchar pixelA;
                std::memcpy(&pixelA, PA + i * pix_sizeA, pix_sizeA);
                double alphaValue = double(pixelA) / 255.0;
                for (int c = 0; c < 3; c++) {
                    // formula : Intensity = alpha * Foreground + (1.0 - alpha) * Background (with alpha in [0..1])
                    //backgroundP[i + pos.x][c] = GetByteInRange(alphaValue * double(foregroundP[i]) + (1.0 - alphaValue) * double(backgroundP[i + pos.x][c]));
                    pixelB[c] = GetByteInRange(alphaValue * double(pixelF[c]) + (1.0 - alphaValue) * double(pixelB[c]));
                }
                std::memcpy(PB + i * pix_sizeB, &pixelB, pix_sizeB);
            }
        }
    }
}

void PasteImageGrayPlusAlphaOnGrayFast(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                                       const cv::Point pos, const bool &multiThread) // paste gray + alpha image on gray background - fast version
{
    // compute limits of image to paste
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    size_t stepA = alpha.step;
    int pix_sizeB = (int)background.elemSize();
    int pix_sizeF = (int)foreground.elemSize();
    int pix_sizeA = (int)alpha.elemSize();
    uchar *ptrB = background.ptr(0);
    const uchar* ptrF = foreground.ptr(0);
    const uchar* ptrA = alpha.ptr(0);

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar *PB = ptrB + (j + pos.y) * stepB + pos.x * pix_sizeB;
            const uchar *PF = ptrF + j * stepF;
            const uchar *PA = ptrA + j * stepA;
            for (int i = minX; i < maxX; i++) {
                uchar pixelB;
                std::memcpy(&pixelB, PB + i * pix_sizeB, pix_sizeB);
                uchar pixelF;
                std::memcpy(&pixelF, PF + i * pix_sizeF, pix_sizeF);
                uchar pixelA;
                std::memcpy(&pixelA, PA + i * pix_sizeA, pix_sizeA);
                double alphaValue = double(pixelA) / 255.0;

                // formula : Intensity = alpha * Foreground + (1.0 - alpha) * Background (with alpha in [0..1])
                pixelB = GetByteInRange(alphaValue * double(pixelF) + (1.0 - alphaValue) * double(pixelB));

                std::memcpy(PB + i * pix_sizeB, &pixelB, pix_sizeB);
            }
        }
    }
}

void PasteImagePlusAlphaFast(cv::Mat &background, const cv::Mat &foreground, const cv::Mat &alpha,
                             const cv::Point &pos, const bool &multiThread) // paste (gray or color) + alpha image on non-alpha BGR background - fast version
{
    if (foreground.channels() == 1) { // image to paste is gray
        if (background.channels() == 1) // destination image is gray
            PasteImageGrayPlusAlphaOnGrayFast(background, foreground, alpha, pos, multiThread);
        else // destination image is BGR
            PasteImageGrayPlusAlphaOnColorFast(background, foreground, alpha, pos, multiThread);
    }
    else if (foreground.channels() == 3) { // image to paste is BGR
        PasteImageColorPlusAlphaOnColorFast(background, foreground, alpha, pos, multiThread);
    }
}

void PasteImageGrayOnGrayOR(cv::Mat &background, const cv::Mat &foreground,
                            const cv::Point &pos, const bool &multiThread) // paste gray image on gray background using a OR combination
{
    // compute limits of image to paste
    int minX = std::max(0, -pos.x);
    int minY = std::max(0, -pos.y);
    int maxX = std::min(foreground.cols, background.cols - pos.x);
    int maxY = std::min(foreground.rows, background.rows - pos.y);

    for (int j = minY; j < maxY; j++) {
        const uchar* foregroundP = foreground.ptr<uchar>(j);
        uchar* backgroundP = background.ptr<uchar>(j + pos.y);
        #pragma omp parallel if(multiThread)
        {
            #pragma omp for nowait
            for (int i = minX; i < maxX; i++) {
                backgroundP[i + pos.x] = foregroundP[i] or backgroundP[i + pos.x];
            }
        }
    }
}

int PasteImageGrayTransparencyWithCount(cv::Mat &background, const cv::Mat &foreground,
                                        const int &originX, const int &originY,
                                        const uchar &transparentColor, const bool &multiThread) // copy gray image onto another with transparency and count the updated pixels
    // the returned count is the number of updated pixels that had a 0 value before
{
    int minX = std::max(0, -originX);
    int minY = std::max(0, -originY);
    int maxX = std::min(foreground.cols, background.cols - originX);
    int maxY = std::min(foreground.rows, background.rows - originY);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    uchar *ptrB = background.ptr<uchar>(0);
    const uchar *ptrF = foreground.ptr<uchar>(0);

    int count = 0;

    #pragma omp parallel if(multiThread)
    {
        #pragma omp for nowait
        for (int j = minY; j < maxY; j++) {
            uchar *PB = ptrB + (j + originY) * stepB + originX;
            const uchar *PF = ptrF + j * stepF;
            for (int i = minX; i < maxX; i++) {
                uchar pixel;
                std::memcpy(&pixel, PF + i, 1);
                if (pixel != transparentColor) { // is it transparent ?
                    uchar value;
                    std::memcpy(&value, PB + i, 1); // test pixel that will be updated
                    if (value == 0) // if pixel to update is zero
                        count++;
                    std::memcpy(PB + i, &pixel, 1);
                }
            }
        }
    }

    return count; // return number of zero-pixels that were updated
}

void PasteImageDoubleFast(cv::Mat &background, const cv::Mat &foreground,
                    const int &originX, const int &originY) // copy 64F image onto another - really fast implementation
{
    // compute limits of image to paste
    int minX = std::max(0, -originX);
    int minY = std::max(0, -originY);
    int maxX = std::min(foreground.cols, background.cols - originX);
    int maxY = std::min(foreground.rows, background.rows - originY);

    size_t stepB = background.step;
    size_t stepF = foreground.step;
    int pix_size = (int)background.elemSize();
    uchar* ptrB = background.ptr(0);
    const uchar* ptrF = foreground.ptr(0);
    int width = (maxX - minX) * pix_size;

    for (int j = minY; j < maxY; j++) {
        uchar* PB = ptrB + (j + originY) * stepB + (minX + originX) * pix_size;
        const uchar* PF = ptrF + j * stepF + minX * pix_size;
        memcpy(PB, PF, width);
    }
}

///////////////////////////////////////////////////////////
//// Alpha channel
///////////////////////////////////////////////////////////

cv::Mat AddAlphaToImage(const cv::Mat &source, const cv::Mat &mask) // add alpha channel to BGR image - if mask is empty, transparency = black (0,0,0)
{
    std::vector<cv::Mat> matChannels;
    cv::split(source, matChannels); // split image in separate channels

    if (mask.empty()) { // no mask ? create one with transparency of source image = black (0,0,0)
        cv::Mat alpha;
        cv::cvtColor(source, alpha, cv::COLOR_BGR2GRAY); // image to gray
        cv::Mat mask = 255 - (alpha == 0); // compute mask
        alpha.setTo(255, mask); // alpha to max where pixels are
        matChannels.push_back(alpha); // add alpha channel to channels stack, contains transparency based on black color
    }
    else {
        matChannels.push_back(mask); // add alpha channel to channels stack, contains provided mask
    }

    cv::Mat result;
    cv::merge(matChannels, result); // merge back all channels + BGR + alpha

    return result;
}

cv::Mat ImageGrid(const int &width, const int &height, const int &interval, const cv::Vec3b color1, const cv::Vec3b color2) // get BGR image with colored blocks
{
    cv::Mat result = cv::Mat(height, width, CV_8UC3);

    for (int j = 0; j < height; j++) {
        cv::Vec3b* resultP = result.ptr<cv::Vec3b>(j);
        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int i = 0; i < width; i++) {
                if ((j % interval >= 0) and (j % interval < interval / 2)) {
                    if ((i % interval >= 0) and (i % interval < interval / 2)) {
                        resultP[i] = color1;
                    }
                    else {
                        resultP[i] = color2;
                    }
                }
                else {
                    if ((i % interval >= 0) and (i % interval < interval / 2)) {
                        resultP[i] = color2;
                    }
                    else {
                        resultP[i] = color1;
                    }
                }
            }
        }
    }

    return result;
}

cv::Mat ImageAlphaWithGrid(const cv::Mat &source, const int &interval, const cv::Vec3b color1, const cv::Vec3b color2) // get BGR image from BGRA with colored blocks where there is transparency
{
    cv::Mat result = ImageGrid(source.cols, source.rows, interval, color1, color2);

    PasteImageAlphaFast(result, source, cv::Point(0, 0), true);

    return result;
}

void ImageToBGRplusAlpha(const cv::Mat &source, cv::Mat &dest, cv::Mat &alpha) // split an image with n channels to BGR + alpha
{
    alpha = cv::Mat(source.rows, source.cols, CV_8UC1);

    if (source.channels() == 1) { // if gray image, no alpha mask -> transform it to 3 channels BGR
        cv::cvtColor(source, dest, cv::COLOR_GRAY2BGR); // convert gray image to BGR
        alpha = 255; // alpha is set to maximum opacity
    }
    else if (source.channels() == 2) { // source is gray + alpha
        std::vector<cv::Mat1b> channels; // split source image
        cv::split(source, channels);

        std::vector<cv::Mat1b> BGR; // merge gray channel 3 times to obtain a BGR image
        BGR.push_back(channels[0]);
        BGR.push_back(channels[0]);
        BGR.push_back(channels[0]);
        cv::merge(BGR, dest);

        alpha = channels[1]; // alpha is from source image
    }
    else if (source.channels() == 3) { // if 3 channels, no alpha mask -> keep image as is
        source.copyTo(dest); // keep image
        alpha = 255; // alpha is set to maximum opacity
    }
    else if (source.channels() == 4) { // source is BGRA
        std::vector<cv::Mat1b> channels; // split source image
        cv::split(source, channels);

        std::vector<cv::Mat1b> BGR; // merge first 3 channels to obtain a BGR image
        BGR.push_back(channels[0]);
        BGR.push_back(channels[1]);
        BGR.push_back(channels[2]);
        cv::merge(BGR, dest);

        channels[3].copyTo(alpha); // alpha is from source image
    }
}

std::vector<cv::Mat> SplitImage(const cv::Mat source) // split image to channels
{
    std::vector<cv::Mat> matChannels;
    cv::split(source, matChannels); // split image in separate channels

    return matChannels;
}

///////////////////////////////////////////////////////////
//// Analysis
///////////////////////////////////////////////////////////

std::vector<double> HistogramImageGray(const cv::Mat &source) // compute histogram of gray image
{
    std::vector<double> histogram;
    histogram.reserve(256);
    for (int i = 0; i < 256; i++)
        histogram.push_back(0.0);

    int total = source.cols * source.rows;
    const uchar* sourceP = source.ptr<uchar>(0);
    #pragma omp parallel for
    for (int n = 0; n < total; n++)
        histogram[sourceP[n]]++;

    return histogram;
}

std::vector<double> HistogramImageGrayWithMask(const cv::Mat &source, const cv::Mat &mask) // compute histogram of gray image using a mask
{
    if (mask.empty()) // no mask = entire image
        return HistogramImageGray(source); // so return entire image histogram

    std::vector<double> histogram;
    histogram.reserve(256);
    for (int i = 0; i < 256; i++)
        histogram.push_back(0.0);

    int total = source.cols * source.rows;
    const uchar* sourceP = source.ptr<uchar>(0);
    const uchar* maskP = mask.ptr<uchar>(0);
    //#pragma omp parallel for // bad idea, order must be maintained !
    for (int n = 0; n < total; n++)
        if (maskP[n] != 0)
            histogram[sourceP[n]]++;

    return histogram;
}

cv::Vec3d MeanWeightedColor(const cv::Mat &source, const cv::Mat &mask) // use an histogram to get mean weighted color of an image area (BGR or BGRA)
{
    double total = cv::countNonZero(mask); // total number of pixels in mask / area

    if (total == 0)
        return cv::Vec3d(0, 0, 0);

    // split image to gray channels
    std::vector<cv::Mat> grayChannels;
    cv::split(source, grayChannels);
    int nbChannels = std::min(3, source.channels()); // no more than 3 channels, this way we can compute gray, color and color+alpha images

    std::vector<std::vector<double>> histograms;
    for (int c = 0; c < nbChannels; c++)
        histograms.push_back(HistogramImageGrayWithMask(grayChannels[c], mask));

    cv::Vec3d result = cv::Vec3d(0, 0, 0);
    for (int c = 0; c < nbChannels; c++)
        for (int n = 0; n < 256; n++)
            result[c] += double(n) * histograms[c][n] / total;

    return result;
}

double MeanWeightedGray(const cv::Mat &source, const cv::Mat &mask) // use an histogram to get mean weighted gray of an image area
{
    double total = cv::countNonZero(mask); // total number of pixels in mask / area

    if (total == 0)
        return 0;

    std::vector<double> histogram = HistogramImageGrayWithMask(source, mask);

    double result = 0;
    for (int n = 0; n < 256; n++)
        result += double(n) * histogram[n] / total;

    return result;
}

cv::Mat CreatePaletteImageFromImage(const cv::Mat3b &source) // parse BGR image and create a one-line RGB palette image from all colors - super-fast !
    // the resulting Mat can be used as palette values
    // it is in RGB format, not BGR !
{
    // get RGB values in array std::set (super-fast !)
    std::set<int> paletteInt;
    for (cv::Vec3b &p : cv::Mat_<cv::Vec3b>(source)) // iterate over pixels
        paletteInt.insert((p[2] << 16) | (p[1] << 8) | (p[0])); // RGB "hatch" representation of the pixel

    int nbColors = paletteInt.size(); // how many colors found ?
    cv::Mat3b palette = cv::Mat::zeros(std::sqrt(nbColors) + 1, std::sqrt(nbColors) + 1, CV_8UC3); // create palette image on one line, nbColors pixels wide

    cv::Vec3b* paletteP = palette.ptr<cv::Vec3b>(0);
    auto value = paletteInt.begin();
    for (int n = 0; n < nbColors; n++) { // parse palette
        paletteP[n] = cv::Vec3b(*value & 0x000000FF, (*value & 0x0000FF00) >> 8, (*value & 0x00FF0000) >> 16); // and set pixel color to palette image
        value++;
    }

    return palette;
}

///////////////////////////////////////////////////////////
//// Image files
///////////////////////////////////////////////////////////

void SavePNG(const std::string &filename, const cv::Mat &source, const bool &transparency) // save PNG with or without transparency
{
    if (!transparency) {
        cv::imwrite(filename, source); // save to PNG image without alpha
    }
    else {
        cv::Mat alpha;
        source.copyTo(alpha);
        alpha = AddAlphaToImage(alpha, cv::Mat());
        cv::imwrite(filename, alpha); // save to PNG image with alpha
    }
}

void SaveFloatImageAsTiff(const cv::Mat &source, const std::string &filename) // save CV_16F floating-point image to TIFF
    // once saved, read float files as : cv::imread("image.tif", cv::IMREAD_ANYDEPTH);
{
    cv::imwrite(filename + ".tif", source, {cv::IMWRITE_TIFF_COMPRESSION, 34925}); // range [ 0.0 .. 1.0 ] - compression is LZMA
}

///////////////////////////////////////////////////////////
//// MATLAB equivalents
///////////////////////////////////////////////////////////

cv::Mat conv2d(const cv::Mat &img, const cv::Mat &kernel, const Conv2DShape &shape) // equivalent to MATLAB conv2d
    // shape :
    //    full - return the full convolution, including border
    //    same - return only the part that corresponds to the original image
    //    valid - return only the submatrix containing elements that were not influenced by the border
{
    cv::Mat dest;
    cv::Mat source = img;

    if(shape == conv2d_full) {
        source = cv::Mat();
        const int additionalRows = kernel.rows - 1;
        const int additionalCols = kernel.cols - 1;
        cv::copyMakeBorder(img, source, (additionalRows + 1) / 2, additionalRows / 2, (additionalCols + 1) / 2, additionalCols / 2, cv::BORDER_CONSTANT, cv::Scalar(0));
    }

    cv::Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
    cv::Mat kernelFlipped;
    cv::flip(kernel, kernelFlipped, -1); // this a convolution so flip the kernel !
    kernelFlipped /= cv::sum(kernelFlipped); // normalize it
    cv::filter2D(source, dest, img.depth(), kernelFlipped, anchor, 0, cv::BORDER_CONSTANT);

    if(shape == conv2d_valid) {
        dest = dest.colRange((kernel.cols - 1) / 2, dest.cols - kernel.cols / 2).rowRange((kernel.rows - 1) / 2, dest.rows - kernel.rows / 2);
    }

  return dest;
}

cv::Mat filter2(const cv::Mat &img, const cv::Mat &kernel) // equivalent to MATLAB filter2
{
    cv::Mat result;

    cv::filter2D(img, result, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT); // the trick is using BORDER_CONSTANT (0-padding)

    return result;
}

///////////////////////////////////////////////////////////
//// Mask utils
///////////////////////////////////////////////////////////

cv::Rect FindMaskContentBoundingRect(const cv::Mat &source) // return the bounding rect of the content of a mask
    // this is simpler : use directly cv::boundingRect on a mask, zero pixels are ignored
{
    assert(source.channels() == 1);

    int minX = source.cols;
    int minY = source.rows;
    int maxX = -1;
    int maxY = -1;

    for (int row = 0; row < source.rows; row++) {
        for (int col = 0; col < source.cols; col++) {
            if (source.at<uchar>(row, col) != 0) {
                if (col < minX)
                    minX = col;
                if (row < minY)
                    minY = row;
                if (col > maxX)
                    maxX = col;
                if (row > maxY)
                    maxY = row;
            }
        }
    }

    if ((maxX < 0) or (maxY < 0) or (minX == source.cols) or (minY == source.rows)) // the mask is empty !
        return cv::Rect(0, 0, -1, -1);
    else
        return cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
}

cv::Mat AdjustMaskToRect(const cv::Mat &source, cv::Rect &rect) // adjust a gray mask to its content
{
    rect = FindMaskContentBoundingRect(source);

    return source(rect).clone();
}

///////////////////////////////////////////////////////////
//// DPI utils
///////////////////////////////////////////////////////////

cv::Size ComputeDPI(const double &diagonalUnits, const cv::Size &aspectRatio, const cv::Size &resolution) // compute DPI from diagonal and aspect ratio
    // diagonalUnits is the diagonal of the rectangle, for example 17.3" or 181cm
    // resolution is (width, height) in pixels
{
    double factor = diagonalUnits / std::sqrt(double(aspectRatio.width * aspectRatio.width + aspectRatio.height * aspectRatio.height)); // factor for computing width and heught

    double widthUnits = double(aspectRatio.width) * factor; // width of the rectangle in diagonal units
    double heightUnits = double(aspectRatio.height) * factor; // height of the rectangle in diagonal units

    cv::Size DPI;
    DPI.width =  std::round(double(resolution.width)  / widthUnits);
    DPI.height = std::round(double(resolution.height) / heightUnits);

    return DPI;
}

cv::Size ComputeDPI(const cv::Size &sizeUnits, const cv::Size &resolution) // compute DPI from width anf height
    // sizeUnits is the dimensions (width, height) of the rectangle, for example (15.3", 8.5") or (50cm, 25cm)
    // resolution is (width, height) in pixels
{
    cv::Size DPI;
    DPI.width  = std::round(double(resolution.width)  / double(sizeUnits.width));
    DPI.height = std::round(double(resolution.height) / double(sizeUnits.height));

    return DPI;
}

///////////////////////////////////////////////////////////
//// Orientation of image
///////////////////////////////////////////////////////////

int GetImageOrientation(const cv::Mat &source) // get image orientation : portrait, landscape or square
    // returned value if from enum orientationImage
{
    // empty image -> error
    if (source.empty())
        return orientation_error;

    // portrait
    if (source.cols < source.rows)
        return orientation_portrait;

    // landscape
    if (source.cols > source.rows)
        return orientation_landscape;

    // square
    return orientation_square;
}

int CompareImagesOrientation(const cv::Mat &source1, const cv::Mat &source2, const bool &ignoreSquares) // compare orientations of two images
    // ignoreSquares indicates if square images are treated differently - if true a square image has same orientation as the other image
    // for image orientation see enum orientationImage
    // returned value is from the enum orientationImageComparison
{
    // empty image -> error
    if ((source1.empty()) or (source2.empty()))
        return orientation_comparison_error;

    // get image orientations
    int orientation1 = GetImageOrientation(source1);
    int orientation2 = GetImageOrientation(source2);

    // one or more images are square ?
    if ((orientation1 == orientation_square) or (orientation2 == orientation_square)) { // if one of the images if square
        if (ignoreSquares) // ignore squares -> orientation is the same
            return orientation_comparison_same;
        else // don't ignore sqaures : return a square orientation
            return orientation_comparison_square;
    }

    // none of the images are square or ignore square images : we can compare the images orientation
    if (orientation1 == orientation2) // same orientation ?
        return orientation_comparison_same;
    else // not the same orientation
        return orientation_comparison_different;
}


