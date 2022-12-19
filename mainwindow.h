/*#-------------------------------------------------
#
#    ImageMatch: Find Image Duplicates with openCV
#
#       by AbsurdePhoton - www.absurdephoton.fr
#
#                   v0 - 2022/12/18
#
#-------------------------------------------------*/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

#include <QMainWindow>
#include <QMessageBox>
#include <QElapsedTimer>
#include <QListWidget>
#include <QTreeWidget>
#include <QStyledItemDelegate>
#include <QPainter>
#include <QScreen>
#include <QMovie>
#include <QWhatsThis>
#include <QDirIterator>

#include <thread>
#include <omp.h>

#include "widgets/file-dialog.h"
#include "lib/image-compare.h"
#include "lib/image-utils.h"
#include "lib/image-transform.h"
#include "lib/image-color.h"
#include "lib/dominant-colors.h"
#include "lib/config-file.h"
#include "lib/string-utils.h"


class TreeWidgetDelegate : public QStyledItemDelegate // delegate to add a line at the bottom of QTreeWidgetItems
{

public:
    explicit TreeWidgetDelegate(QObject * parent = nullptr) : QStyledItemDelegate(parent) { } // delegate definition

    void paint(QPainter * painter, const QStyleOptionViewItem & option, const QModelIndex & index ) const // replace paint event
    {
        painter->setPen(QColor(Qt::black)); // set color to black
        painter->drawLine(QLine(0, option.rect.bottom(), option.rect.right(), option.rect.bottom())); // draw bottom line under the item
        QStyledItemDelegate::paint(painter, option, index);  // then call normal paint event
    }
};

class ListWidgetDelegate : public QStyledItemDelegate // delegate to add a border to QListItemWidgets
{

public:
    explicit ListWidgetDelegate(QObject *parent = 0)  : QStyledItemDelegate(parent) { } // delegate definition

    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const // replace paint event
    {
        painter->setPen(QColor(Qt::black)); // set color to black
        painter->drawRect(option.rect); // draw border
        QStyledItemDelegate::paint(painter, option, index);  // then call normal paint event
    }
};

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public slots:
    void ImagesListClick(QListWidgetItem *item); // click on item in images list -> toggle checked status
    void ImagesListDoubleClick(QListWidgetItem *item); // double-click on item of images list -> show image in new window
    void DuplicatesListClick(QTreeWidgetItem *item, int column); // click anywhere (except the image) on duplicate item to check it
    void DuplicatesListDoubleClick(QTreeWidgetItem *item, int column); // double-click duplicate image to open it in a new window

/*// define isnan if fast_math is activated (isnan doesn't work with fast_math
#if defined __FAST_MATH__
#   undef isnan
#endif
#if !defined isnan
#   define isnan isnan
#   include <stdint.h>
static inline int isnan(float f)
{
    union { float f; uint32_t x; } u = { f };
    return (u.x << 1) > 0xff000000u;
}
#endif*/

private slots:

    //// GUI
    void on_button_quit_clicked(); // quit GUI
    void on_button_whats_this_clicked(); // What's this function
    void on_comboBox_algo_currentIndexChanged(int algo); // change algorithm -> set value to threshold
    void on_comboBox_level_currentIndexChanged(int level); // set value to threshold doubleSpinBox when level has changed
    void SetCombinedActivated(const imageSimilarityAlgorithm &algorithm, const bool &activated); // activate an algorithm for combined score in options tab
    void HideCombinedAlgorithm(const imageSimilarityAlgorithm &algo, const bool &value); // show/hide algorithm in options tab

    //// Connectors
    /// Images list
    // List management
    void on_button_add_images_clicked(); // button pressed -> add images to list
    void on_button_images_clear_clicked(); // button pressed -> clear images list (and also duplicates list)
    void on_button_images_uncheck_all_clicked(); // button pressed -> uncheck all images in images list
    void on_button_images_check_all_clicked(); // button pressed -> check all images in images list
    void on_button_images_uncheck_all_selected_clicked(); // button pressed -> uncheck selected images in images list
    void on_button_images_check_all_selected_clicked(); // button pressed -> check selected images in images list
    void on_button_images_check_invert_clicked(); // button pressed -> invert check state in images list
    void on_button_images_check_text_clicked(); // button pressed -> find words and check images in images list
    // Remove and delete
    void RemoveImageFromListImages(QListWidgetItem *item); // remove one item from image list by its pointer
    void on_button_images_hide_clicked(); // button pressed -> hide checked images from list
    void on_button_images_hide_error_clicked(); // button pressed -> hide images with errors from list
    void on_button_images_delete_clicked(); // button pressed -> delete images from list
    //// Options
    void on_spinBox_thumbnails_size_valueChanged(int size); // change thumbnails size
    void on_spinBox_reduced_size_valueChanged(int size); // change working image size
    void on_spinBox_nb_features_valueChanged(int nb); // change number of features to find for features detection and homography
    void on_doubleSpinBox_threshold_valueChanged(double nb); // change threshold
    /// duplicates list
    // clear
    void ClearDuplicates(); // clear duplicates list and associated GUI elements and variables + show duplicates count
    // Compute similarities
    void on_button_compare_images_clicked(); // compare images from images list
    // Check
    void on_button_duplicates_uncheck_all_clicked(); // button pressed -> uncheck all duplicates in duplicates list
    void on_button_duplicates_check_all_clicked(); // button pressed -> check all duplicates in duplicates list
    void on_button_duplicates_check_invert_clicked(); // button pressed -> invert all checks in duplicates list
    void on_button_duplicates_check_first_clicked(); // button pressed -> check first duplicate in each batch in duplicates list
    void on_button_duplicates_check_rest_clicked(); // button pressed -> check all duplicates in each group but first one in duplicates list
    void on_button_duplicates_check_text_clicked(); // button pressed -> find words and check duplicates in duplicates list
    // Remove copy and delete
    void on_button_duplicates_hide_clicked(); // button pressed -> remove checked items in duplicates list
    void on_button_duplicates_delete_clicked(); // button pressed -> remove checked items in duplicates list and delete the image file
    void on_button_duplicates_copy_clicked(); // button pressed -> copy files of checked duplicates in duplicates list to another folder
    void on_button_duplicates_move_clicked(); // button pressed -> move checked items in duplicates list to another folder
    // save results
    void on_button_duplicates_save_results_clicked(); // button to save results - for debugging purpose only - deactivate it for final version
    // Tests - deactivate them for final version
    void on_button_test_images_clicked();
    void on_button_examine_clicked(); // set as a test at the beginning, this function is useful to estimate the real visual similarity between two images

    //// sliders and values
    // none for now !

private:
    Ui::MainWindow *ui;

    //// OMP
    int processor_threads; // number of processors to use in parallel tasks (set it to #of machine CPUs - 1)

    //// files
    std::string basefile, basedir, basedirinifile; // for file dialogs : filename, directory and filename without extension

    //// timer and progress
    QElapsedTimer progressTimer; // elapsed time
    qint64 oldProgressTimer; // to know how much has passed since the timer was set
    enum timer_state{timer_stopped, timer_ready, timer_running, timer_finished}; // 4 states for timer
    enum progress{progress_stop, progress_prepare, progress_run, progress_update, progress_finished}; // 5 steps in progress
    int progressStep; // to show intermediate progress
    QLabel *waitWindow; // waiting window
    QMovie *waitMovie; // waiting movie
    QPixmap waitPixmap; // waiting pixmap which will fill movie
    bool stop;

    //// file delete to trash
    bool trashBinEnabled; // if a trash bin is available in the OS
    QString deleteFileMessage; // for displaying messages : indicate if files will be deleted or moved to trash

    //// variables
    // images list
    struct struct_image_info { // all the information one image needs to be processed !
        // state
        bool newImage; // indcates if image was just added
        bool deleted; // indcates if image is not active
        bool error; // indicates image exists but not readable (shown in images list as error)
        // filename info
        std::string fullPath; // full path of image
        std::string folder; // folder of image
        std::string basename; // base file name of image
        std::string extension; // file extension of image
        // image useful information
        std::string type; // file type of image (png, tiff, etc)
        std::string loadwith; // method to read file : with OpenCV or Qt (some file types are not loadable with OpenCV)
        int width; // image width
        int height; // image height
        int imageSize; // image size in pixels (= width x height)
        // clustering
        bool used; // indicates image was already used in duplicates list - used for clustering
        std::vector<int> duplicates; // list of duplicate images for this image
        int group; // an image is member of this group number
        // hashes and points data - cache
        cv::Mat hashTmp; // used for common hashes functions
        cv::Mat hashDNN; // used for DNN result
        std::vector<cv::KeyPoint> keypoints; // features and homography
        cv::Mat descriptors; // features and homography
        std::vector<cv::Vec3d> dominantColors;
        // miniature cache
        QPixmap icon; // icon for display in Qt (images list and duplicates)
        cv::Mat imageReduced; // color miniature
        cv::Mat imageReducedGray; // gray miniature
        // images list and duplicates entry
        QListWidgetItem *imageItem;
        QTreeWidgetItem *duplicateItem;
    };
    std::vector<struct_image_info> images; // contains images list to test

    // duplicates
    struct struct_scores { // for one pair of images, keep the computed information - +1 because img_similarity_count is used for combined score
        float score[static_cast<int>(img_similarity_count + 1)] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; // float values
    };

    struct struct_comparePoints // needed for sorting cv::Point
    {
        bool operator () (const cv::Point &a, const cv::Point &b) const
        {
            return (a.x < b.x) || (a.x == b.x && a.y < b.y); // sort by x then y value of point
        }
    };

    std::map<cv::Point, struct_scores, struct_comparePoints> pairs; // list of all tested images pairs
    std::vector<std::vector<int>> groups; // contains clustered images - each image has a group, the index is the same for both (image #)

    struct struct_compareSimilarities // needed for sorting image similarities enum
    {
        bool operator () (const imageSimilarityAlgorithm &a, const imageSimilarityAlgorithm &b) const
        {
            return (a < b) || (a == b && a < b); // sorted ascending
        }
    };

    std::map<imageSimilarityAlgorithm, std::vector<float>, struct_compareSimilarities> thresholds; // treshold values for each category for each algorithm - read from a config file
    // categories : 0 < dissimilar < different < similar < ∞ (exact)

    // DNN
    cv::dnn::Net dnn; // DNN is only defined (and loaded) once

    // options
    int thumbnailsSize;
    int reducedSize;
    int nbFeatures;
    float threshold;
    imageSimilarityAlgorithm similarityAlgorithm = img_similarity_checksum;

    // delegates for QTreeWidget and QListWidget
    TreeWidgetDelegate *treeWidgetDelegate;
    ListWidgetDelegate *listWidgetDelegate;

    //// UI functions
    // General
    void InitializeValues(); // initialize GUI and variables

    // Progress
    void ProgressInit(); // initialize progress use
    void ShowTimerMessage(const int &state); // set timer message
    void ChangeTimerState(const int &state); // set timer color
    void StopButton(); // step global variable stop to true
    void ShowProgressWait(); // define and show waiting animated icon in modal window
    void HideProgressWait(); // hide waiting animation in modal window
    void ShowProgress(const int &state, const QString message = "", int value = 0, int maximum = 0);

    // files deletion
    void TrashBinTest(); // test file deletetion to trash bin and set global variables to indicate file deletion method
    bool DeleteFile(const std::string &filePath); // delete a file to trash bin or permanently

    //// save & load functions
    void ChangeBaseDir(QString filename); // set base dir and file
    void SaveDirBaseFile(); // keep last open dir
    QPixmap LoadImagePix(const std::string &path, const std::string &engine); // return a Qt QPixmap from image file using different loading engines
    cv::Mat LoadImageMat(const std::string &path, const std::string &engine); // return an OpenCV Mat from image file using different loading engines

	//// Other functions here
    // images list
    QColor ImageTypeColor(const std::string &extension); // returns image file type color
    void PopulateImagesList(const std::string &folder, const bool &recursive); // parse a directory and add images
    void CleanImagesList(); // clean/delete all duplicates in images list
    void ComputeImagesListInfo(); // compute all other required info in images list
    void ShowImagesList(); // display images list from images information
    void ShowImagesListCount(); // show number of images in images list in GUI
    // duplicates list
    cv::Point OrderedPair(const int &i, const int &j); // returns a pair of integer (image index) - pairs must have im1 <= im2
    int GetClosestNeighbourNotUsed(const int &imageNumber, const imageSimilarityAlgorithm &algo); // get closest neighbour's imageNumber of an imageNumber from duplicates of an image
    int GetClosestNeighbour(const int &imageNumber, const imageSimilarityAlgorithm &algo); // get closest imageNumber of a imageNumber from duplicates of an image
    int GetClosestNeighbourHasGroup(const int &imageNumber, const imageSimilarityAlgorithm &algo); // get closest imageNumber of a imageNumber that has members in its duplicates list
    float GetScore(const int &im1, const int &im2, const imageSimilarityAlgorithm &algo); // get algo distance/score between 2 images
    void AddImageToGroup(const int &group, const int &image); // add an image to a group
    void AddImageToGroupCheckScores(const int &group, const int &image, const imageSimilarityAlgorithm &algo); // add an image to a group, checking a minimum score
    int GetLevelFromScore(const imageSimilarityAlgorithm &similarityAlgorithm, const float &score); // get level of a score from thresholds list - categories : 0 < dissimilar < different < similar < ∞ (exact)
    float CombinedScore(const int &i, const int &j); // get combined score for 2 images from previous tests
    bool ImagesAreDuplicates(const int &i, const int &j, const imageSimilarityAlgorithm &similarityAlgorithm, const float &threshold, float &similarity); // compare a pair of images using an algorithm
    std::string GetHashString(const int &imageNumber, const imageSimilarityAlgorithm &similarityAlgorithm); // get hash string from image hash (debug purpose only)
    void CompareImages(); // compare images in images list
    QTreeWidgetItem* GetDuplicateItem(const int &ref); // get a prepared duplicate item for the duplicates view
    void ShowDuplicatesList(); // display alll duplicates : cluster in groups then show the list
    int ShowDuplicatesListCount(); // show number of images in duplicates list
    void SaveResults(); // save results to csv file - for testing purpose only
    // config files
    bool ReadThresholdsConfig(); // read thresholds from config file, return success as bool

};

#endif // MAINWINDOW_H
