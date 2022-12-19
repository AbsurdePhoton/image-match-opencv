/*#-------------------------------------------------
#
#    ImageMatch: Find Image Duplicates with openCV
#
#       by AbsurdePhoton - www.absurdephoton.fr
#
#                   v0 - 2022/12/18
#
#-------------------------------------------------*/


#include "mainwindow.h"
#include "ui_mainwindow.h"


/////////////////// Window init //////////////////////

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //// numeric dot vs comma
    std::setlocale(LC_NUMERIC,"C"); // dot as numeric separator for decimal values - some countries use commas

    //// number of processors for parallel processing
    processor_threads = std::thread::hardware_concurrency(); // find how many processor threads in the system
    if (processor_threads >= 2) // at least 2 processors ?
        processor_threads -= 1; // set max number of threads = nb processors - 1

    omp_set_num_threads(processor_threads); // set usable threads for OMP

    //// opencl
    cv::ocl::setUseOpenCL(false); // deactivate openCL
    /*if (!cv::ocl::haveOpenCL())
    {
        qDebug() << "OpenCL is not available...";
        //return;
    }

    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_ALL))
    {
        qDebug() << "Failed creating the context...";
        //return;
    }

    qDebug() << context.ndevices() << " devices are detected."; //This bit provides an overview of the OpenCL devices you have in your computer
    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        qDebug() << "name:              " << QString::fromStdString(device.name());
        qDebug() << "available:         " << QVariant(device.available()).toString();
        qDebug() << "imageSupport:      " << QVariant(device.imageSupport()).toString();
        qDebug() << "OpenCL_C_Version:  " << QString::fromStdString(device.OpenCL_C_Version());
        qDebug();
    }

    //cv::UMat test = imread("dvdff", cv::IMREAD_COLOR).getUMat(cv::ACCESS_READ);
    //cv::resize(test, test, cv::Size(40, 40));

    cv::ocl::Device(context.device(0)); //Here is where you change which GPU to use (e.g. 0 or 1)*/

    //// GUI
    // Main Window : bars, buttons, etc
    QGuiApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::Round);
    setWindowFlags(Qt::Window | Qt::CustomizeWindowHint | Qt::WindowMinMaxButtonsHint); // just minimize and size buttons
    this->setWindowState(Qt::WindowMaximized); // maximize window
    setFocusPolicy(Qt::StrongFocus); // catch keyboard and mouse in priority
    //statusBar()->setVisible(false); // no status bar
    setWindowIcon(QIcon(":/icons/compare-images.png")); // icon in title bar

    // combobox
    // hash algorithms
    ui->comboBox_algo->blockSignals(true);
    ui->comboBox_algo->addItem("Checksum");
    //ui->comboBox_algo->addItem("aHash");
    ui->comboBox_algo->addItem("pHash");
    ui->comboBox_algo->addItem("dHash");
    ui->comboBox_algo->addItem("idHash");
    //ui->comboBox_algo->addItem("visHash");
    ui->comboBox_algo->addItem("Block Mean");
    ui->comboBox_algo->addItem("Marr Hildreth");
    ui->comboBox_algo->addItem("Radial Variance");
    //ui->comboBox_algo->addItem("Color Moments");
    ui->comboBox_algo->addItem("Dominant colors");
    ui->comboBox_algo->addItem("DNN Classify");
    ui->comboBox_algo->addItem("Features");
    ui->comboBox_algo->addItem("Homography");
    ui->comboBox_algo->addItem("Combined");
    ui->comboBox_algo->insertSeparator(1); // below checksum
    ui->comboBox_algo->insertSeparator(5); //
    ui->comboBox_algo->insertSeparator(9);
    ui->comboBox_algo->insertSeparator(12);
    ui->comboBox_algo->insertSeparator(15);
    ui->comboBox_algo->setCurrentIndex(2);
    ui->comboBox_algo->blockSignals(false);
    ui->comboBox_level->blockSignals(true); // block signals for combobox
    ui->comboBox_level->addItem("Dissimilar");
    ui->comboBox_level->addItem("Different");
    ui->comboBox_level->addItem("Similar");
    ui->comboBox_level->addItem("Exact");
    ui->comboBox_level->setCurrentIndex(2); // set dummy value to combobox
    ui->comboBox_level->blockSignals(false); // restore signals listening to combobox

    // images list
    ui->treeWidget_duplicates->setWordWrap(true); // text of items is word-wrapped
    ui->treeWidget_duplicates->setColumnWidth(0, 50); // set columns size
    ui->treeWidget_duplicates->setColumnWidth(1, 200);
    ui->treeWidget_duplicates->setColumnWidth(2, 70);
    ui->treeWidget_duplicates->setColumnWidth(3, 100);
    ui->treeWidget_duplicates->setColumnWidth(4, 70);
    ui->treeWidget_duplicates->setColumnWidth(5, 500);
    for (int i = 0; i < 6; i++) // set alignment for all columns
        ui->treeWidget_duplicates->header()->setDefaultAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

    // options tab
    for (int i = img_similarity_checksum; i < img_similarity_count; i++) { // hide all algorithms in options tab
        HideCombinedAlgorithm(static_cast<imageSimilarityAlgorithm>(i), true);
        SetCombinedActivated(static_cast<imageSimilarityAlgorithm>(i), false);
    }
    ui->frame_group_thumbnails_size->setDisabled(false); // enable options
    ui->frame_group_reduced_size->setDisabled(false);

    // test buttons
    ui->button_test_images->setHidden(true);
    ui->doubleSpinBox_test_images->setHidden(true);

    // slots : for mouse clicks on images list and duplicates list
    connect(ui->listWidget_image_list, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(ImagesListClick(QListWidgetItem*))); // single click on images list
    connect(ui->listWidget_image_list, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(ImagesListDoubleClick(QListWidgetItem*))); // double click on images list
    connect(ui->treeWidget_duplicates, SIGNAL(itemClicked(QTreeWidgetItem*, int)), this, SLOT(DuplicatesListClick(QTreeWidgetItem*, int))); // single click on duplicates list
    connect(ui->treeWidget_duplicates, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)), this, SLOT(DuplicatesListDoubleClick(QTreeWidgetItem*, int))); // double click on duplicates list

    // initial variable values
    InitializeValues(); // intial variable values and other GUI elements

    // draw supplemental lines on child widgets in QTreeWidget (duplicates list) and QListWidget (images list)
    treeWidgetDelegate = new TreeWidgetDelegate; // new delegate
    ui->treeWidget_duplicates->setItemDelegate(treeWidgetDelegate); // use this delegate
    listWidgetDelegate = new ListWidgetDelegate; // same here
    ui->listWidget_image_list->setItemDelegate(listWidgetDelegate);

    // initial values for options and combobox
    ui->comboBox_algo->setCurrentIndex(0); // should also set threshold
    ui->comboBox_level->setCurrentIndex(3); // "Similar"
    ui->spinBox_thumbnails_size->setValue(176); // thumbnails size default is 176x176px
    ui->spinBox_reduced_size->setValue(256); // default working images size is 256x256px
    ui->spinBox_nb_features->setValue(150); // default number of image features to find (also for homography algorithm)

    //qDebug("Variables initialized");
    //qDebug() << QImageReader::supportedImageFormats(); // show in debug what images OpenCV can read
}

MainWindow::~MainWindow() // Main window destructor
{
    delete ui;
}


/////////////////// for switch with string values //////////////////////

namespace fnv
{
  constexpr uint64_t _(uint64_t h, const char* s)
  {
    return (*s == 0) ? h :
      _((h * 1099511628211ull) ^ static_cast<uint64_t>(*s), s+1); // use a hash
  }
}

constexpr uint64_t _(const char* s)
{
  return fnv::_(14695981039346656037ull, s); // use a hash
}

uint64_t _(const std::string& s)
{
  return fnv::_(14695981039346656037ull, s.data()); // use a hash
}


///////////////////      GUI       //////////////////////

/// General

void MainWindow::InitializeValues() // Global variables and GUI elements init
{
    //// files loading and saving default folder and filename
    basedirinifile = QDir::currentPath().toUtf8().constData(); // where the folder ini file is stored
    basedirinifile += "/dir.ini"; // file options file name
    cv::FileStorage fs(basedirinifile, cv::FileStorage::READ); // open dir ini file
    if (fs.isOpened()) { // success ?
        fs["BaseDir"] >> basedir; // load last directory value
    }
        else basedir = "/home/"; // default base path and file
    basefile = "example";

    //// other GUI items
    // none

    //// file delete to trash bin test
    TrashBinTest(); // test if OS has a trash bin

    //// Timer
    ProgressInit(); // initialize progress bar
    ShowProgress(progress_stop); // progress bar is animated to show inactivity

    //// lists
    images.clear(); // list of loaded images
    pairs.clear(); // list of images pairs scores

    //// config files
    // thresholds config file
    for (int i = img_similarity_checksum; i < img_similarity_count + 1; i++) {
        std::vector<float> v;
        for (int j = 0; j < 4; j++)
            v.push_back(0);
        thresholds.insert(std::make_pair(static_cast<imageSimilarityAlgorithm>(i), v));
    }
    ReadThresholdsConfig(); // read thresholds from file
}

void MainWindow::on_comboBox_algo_currentIndexChanged(int algo) // change algo -> set value to threshold
{
    std::string text = ui->comboBox_algo->currentText().toStdString(); // current text from combobox
    similarityAlgorithm = img_similarity_count; // default similarity algorithm

    // set current similarity algorithm from text value
    switch (_(text)) { // test text value
        case _("Checksum"): // element value from combobox
            similarityAlgorithm = img_similarity_checksum; // set corresponding similarity algorithm value
            break; // exit
        case _("pHash"): // the same for all the next values
            similarityAlgorithm = img_similarity_pHash;
            break;
        case _("dHash"):
            similarityAlgorithm = img_similarity_dHash;
            break;
        case _("idHash"):
            similarityAlgorithm = img_similarity_idHash;
            break;
        case _("Block Mean"):
            similarityAlgorithm = img_similarity_block_mean;
            break;
        case _("Marr Hildreth"):
            similarityAlgorithm = img_similarity_marr_hildreth;
            break;
        case _("Radial Variance"):
            similarityAlgorithm = img_similarity_radial_variance;
            break;
        /*case _("Color Moments"):
            similarityAlgorithm = img_similarity_color_moments;
            break;*/
        case _("Dominant colors"):
            similarityAlgorithm = img_similarity_dominant_colors;
            break;
        case _("Features"):
            similarityAlgorithm = img_similarity_features;
            break;
        case _("Homography"):
            similarityAlgorithm = img_similarity_homography;
            break;
        case _("DNN Classify"):
            similarityAlgorithm = img_similarity_dnn_classify;
            break;
        case _("Combined"): // special
            similarityAlgorithm = img_similarity_count;
            break;
        default:
            similarityAlgorithm = img_similarity_count;
    }

    ui->doubleSpinBox_threshold->setValue(thresholds[similarityAlgorithm][ui->comboBox_level->currentIndex()]); // set current threshold value for this similarity algorithm
}

void MainWindow::on_comboBox_level_currentIndexChanged(int level) // set value to threshold doubleSpinBox when level has changed
{
    ui->doubleSpinBox_threshold->setValue(thresholds[similarityAlgorithm][level]);
}

void MainWindow::SetCombinedActivated(const imageSimilarityAlgorithm &algorithm, const bool &activated) // activate an algorithm for combined score in options tab
{
    switch (algorithm) {
        case img_similarity_checksum:           ui->checkBox_combined_checksum->setChecked(activated);break;
        case img_similarity_pHash:              ui->checkBox_combined_phash->setChecked(activated);break;
        case img_similarity_dHash:              ui->checkBox_combined_dhash->setChecked(activated);break;
        case img_similarity_idHash:             ui->checkBox_combined_idhash->setChecked(activated);break;
        case img_similarity_block_mean:         ui->checkBox_combined_blockmean->setChecked(activated);break;
        case img_similarity_marr_hildreth:      ui->checkBox_combined_marrhildreth->setChecked(activated);break;
        case img_similarity_radial_variance:    ui->checkBox_combined_radialvariance->setChecked(activated);break;
        //case img_similarity_color_moments:      ui->checkBox_combined_colormoments->setChecked(activated);break;
        case img_similarity_dominant_colors:    ui->checkBox_combined_dominantcolors->setChecked(activated);break;
        case img_similarity_features:           ui->checkBox_combined_features->setChecked(activated);break;
        case img_similarity_homography:         ui->checkBox_combined_homography->setChecked(activated);break;
        case img_similarity_dnn_classify:       ui->checkBox_combined_classify->setChecked(activated);break;
    }
}

void MainWindow::HideCombinedAlgorithm(const imageSimilarityAlgorithm &algo, const bool &value) // show/hide algorithm in options tab
{
    switch (algo) {
        case img_similarity_checksum:           ui->frame_combined_checksum->setDisabled(value);break;
        case img_similarity_pHash:              ui->frame_combined_phash->setDisabled(value);break;
        case img_similarity_dHash:              ui->frame_combined_dhash->setDisabled(value);break;
        case img_similarity_idHash:             ui->frame_combined_idhash->setDisabled(value);break;
        case img_similarity_block_mean:         ui->frame_combined_blockmean->setDisabled(value);break;
        case img_similarity_marr_hildreth:      ui->frame_combined_marrhildreth->setDisabled(value);break;
        case img_similarity_radial_variance:    ui->frame_combined_radialvariance->setDisabled(value);break;
        //case img_similarity_color_moments:      ui->frame_combined_colormoments->setDisabled(value);break;
        case img_similarity_dominant_colors:    ui->frame_combined_dominantcolors->setDisabled(value);break;
        case img_similarity_features:           ui->frame_combined_features->setDisabled(value);break;
        case img_similarity_homography:         ui->frame_combined_homography->setDisabled(value);break;
        case img_similarity_dnn_classify:       ui->frame_combined_classify->setDisabled(value);break;
    }
}

/// Progress

void MainWindow::ProgressInit() // initialize progress use
{
    waitMovie = new QMovie(":/icons/wait.gif"); // set movie to gif file
    waitPixmap = QPixmap(":/icons/wait.gif"); // this one is for getting the dimensions when needed
}

void MainWindow::ShowTimerMessage(const int &state) // show timer message according to action state
{
    if (state == timer_stopped) { // reset
        ui->timer->display("---------"); // reset timer
    }
    if (state == timer_ready) { // something will be computed = yellow
        ui->timer->display("..BUSY..."); // wait message
    }
    else if (state == timer_running) { // running, show elapsed time = green
        int milliseconds = int(progressTimer.elapsed() % 1000); // milliseconds
        int seconds = int(progressTimer.elapsed() / 1000); // seconds
        ui->timer->display(QString("%1").arg(seconds, 7, 10, QChar(' '))
                          + "."
                          + QString("%1").arg(milliseconds / 100, 1, 10, QChar('0'))); // show value in LCD timer (ten characters)
    }
}

void MainWindow::ChangeTimerState(const int &state) // set timer color according to action state
{
    if (state == timer_stopped) { // stopped = red
        QPalette pal = ui->timer->palette(); // use a palette
        pal.setColor(QPalette::Normal, QPalette::Light, QColor(255,0,0)); // set values for QLCDNumber
        pal.setColor(QPalette::Normal, QPalette::WindowText, QColor(255,92,92));
        pal.setColor(QPalette::Normal, QPalette::Dark, QColor(164,0,0));
        ui->timer->setPalette(pal); // set palette to QLCDNumber
    }
    else if (state == timer_ready) { // something will be computed = orange
        QPalette pal = ui->timer->palette(); // use a palette
        pal.setColor(QPalette::Normal, QPalette::Light, QColor(255,128,0)); // set values for QLCDNumber
        pal.setColor(QPalette::Normal, QPalette::WindowText, QColor(255,128,92));
        pal.setColor(QPalette::Normal, QPalette::Dark, QColor(164,82,0));
        ui->timer->setPalette(pal); // set palette to QLCDNumber
    }
    else if (state == timer_running) { // running, show elapsed time = yellow
        QPalette pal = ui->timer->palette(); // use a palette
        pal.setColor(QPalette::Normal, QPalette::Light, QColor(255,255,0)); // set values for QLCDNumber
        pal.setColor(QPalette::Normal, QPalette::WindowText, QColor(255,255,92));
        pal.setColor(QPalette::Normal, QPalette::Dark, QColor(164,164,0));
        ui->timer->setPalette(pal); // set palette to QLCDNumber
    }
    else if (state == timer_finished) { // finished and happy = green
        QPalette pal = ui->timer->palette(); // use a palette
        pal.setColor(QPalette::Normal, QPalette::Light, QColor(0,255,0)); // set values for QLCDNumber
        pal.setColor(QPalette::Normal, QPalette::WindowText, QColor(92,255,92));
        pal.setColor(QPalette::Normal, QPalette::Dark, QColor(0,164,0));
        ui->timer->setPalette(pal); // set palette to QLCDNumber
    }
}

void MainWindow::StopButton() // step global variable stop to true
{
    stop = true;
}

void MainWindow::ShowProgressWait() // define and display waiting animated icon in modal window
{
    QSize screenSize = qApp->screens()[0]->size(); // get current screen size

    waitWindow = new QLabel(this); // the wait window is just a QLable
    waitWindow->setWindowModality(Qt::WindowModal); // window is modal
    waitWindow->setWindowFlags(Qt::Window | Qt::FramelessWindowHint | Qt::SplashScreen); // window has no frame and is a splash screen
    waitWindow->setWindowTitle("Please be patient..."); // window title
    waitWindow->setGeometry((screenSize.width() - waitPixmap.width()) / 2, (screenSize.height() - waitPixmap.height()) / 2, waitPixmap.width(), waitPixmap.height()); // window/widget size
    waitWindow->setMovie(waitMovie); // tell that this window/widget is animated
    waitWindow->setScaledContents(true); // content is scaled to window size

    QPushButton *button = new QPushButton(waitWindow);
    button->setGeometry(10, 10, 36, 36);
    button->setIcon(QIcon(":/icons/stop.png"));
    button->setIconSize(QSize(32, 32));
    button->setStyleSheet("QPushButton {\n	background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n stop: 0 #FFFFFF, stop: 1 #E0E0E0);\n	border-radius: 10px;\n	border: 2px outset #8f8f91;\n	color rgb(0,0,0);\n}\nQPushButton:pressed {\n	border: 2px inset #8f8f91;\n}\nQToolTip {\n    border:2px solid black;\n	padding:5px;\n	background-color:rgb(64,64,64);\n	color:white;\n	font-size: 14px;\n}");
    stop = false;
    connect(button, &QPushButton::released, this, &MainWindow::StopButton);

    waitWindow->show(); // show the window
    waitMovie->start(); // start animation
}

void MainWindow::HideProgressWait() // hide waiting animation in modal window
{
    waitMovie->stop(); // stop animation
    waitWindow->close(); // close modal window
}

void MainWindow::ShowProgress(const int &state, const QString message, int value, int maximum) // show/update progress bar and messages according to action state
{
    if (state == progress_stop) { // stop all : timer red showing "---", progress moving bar
        QApplication::restoreOverrideCursor(); // Restore cursor
        ChangeTimerState(timer_stopped);
        ShowTimerMessage(timer_stopped);
        ui->progress->setMaximum(0);
        ui->progress->setValue(0);
        ui->doubleSpinBox_rate->setValue(0);
        //ShowResults();
    }
    else if (state == progress_prepare) { // timer orange showing "busy", progress moving bar
        QApplication::setOverrideCursor(Qt::WaitCursor); // wait cursor
        ChangeTimerState(timer_ready);
        ShowTimerMessage(timer_ready);
        ui->progress->setMaximum(0);
        ui->progress->setValue(0);
        ui->doubleSpinBox_rate->setValue(0);

        ShowProgressWait(); // show animated waiting icon

        //stop = false;
    }
    else if (state == progress_run) { // timer yellow showing elapsed time, message required here, progress set to value
        progressTimer.start();
        oldProgressTimer = progressTimer.elapsed();
        ChangeTimerState(timer_running);
        ShowTimerMessage(timer_running);
        ui->progress->setMaximum(maximum);
        if (message != "")
            ui->label_operation->setText(message);
        ui->doubleSpinBox_rate->setValue(0);
    }
    else if (state == progress_update) { // just update timer value and progress, message optional
        if (progressTimer.elapsed() - oldProgressTimer > 1000) { // more than 1s elapsed
            ShowTimerMessage(timer_running);
            ui->progress->setValue(value);
            if (message != "")
                ui->label_operation->setText(message);
            ui->doubleSpinBox_rate->setValue(double(value) / (progressTimer.elapsed() / 1000.0)); // compute operation rate (per second) and show it
            //ShowResults();
            oldProgressTimer = progressTimer.elapsed();
        }
    }
    else if (state == progress_finished) { // timer green showing last value, progress to maximum, message required here
        QApplication::restoreOverrideCursor(); // Restore cursor
        ChangeTimerState(timer_finished);
        ShowTimerMessage(timer_running);
        ui->progress->setValue(ui->progress->maximum());
        if (message != "")
            ui->label_operation->setText(message);
        ui->doubleSpinBox_rate->setValue(double(ui->progress->maximum()) / (progressTimer.elapsed() / 1000.0)); // compute final operation rate (per second) and show it
        HideProgressWait(); // hide animated waiting icon
        //ShowResults();
        //stop = false;
    }

    qApp->processEvents();
}

//// Delete files ////

void MainWindow::TrashBinTest() // test file deletion to trash bin and set global variables to indicate file deletion method
{
    // delete to trash bin test
    std::ofstream output("test.tst"); // create a dummy file
    QString pathInTrash = "";
    trashBinEnabled = QFile::moveToTrash("test.tst", &pathInTrash); // test file deletion with return value pathInTrash for the dummy file

    // end of test
    if (trashBinEnabled) { // trash bin found ?
        deleteFileMessage = "to Trash Bin"; // part of future messages
        if (pathInTrash != "") // dummy file path in trash is not null
            QFile::remove(pathInTrash); // delete it from the trash (permanent delete)
    }
    else { // no trash bin found ! dummy file has been permanently deleted
        deleteFileMessage = "permanently"; // part of future messages
    }
}

bool MainWindow::DeleteFile(const std::string &filePath) // delete a file to trash bin or permanently
{
    if (trashBinEnabled) // OS trash bin found ?
        return QFile::moveToTrash(QString::fromStdString(filePath)); // move file to delete to trash
    else // no trash bin -> delete file permanently
        return (std::remove(filePath.c_str()) == 0);
}


//// Events ////

/// General

void MainWindow::on_button_whats_this_clicked() // What's this function
{
    QWhatsThis::enterWhatsThisMode(); // enter what's this? mode
}

void MainWindow::on_button_quit_clicked() // quit GUI
{
    int quit = QMessageBox::question(this, "Quit this wonderful program", "Are you sure you want to quit?", QMessageBox::Yes|QMessageBox::No); // quit, are you sure ?
    if (quit == QMessageBox::No) // don't quit !
        return;

    // release global variables memory initialized with "new"
    delete waitMovie; // free animated wainting icon
    delete listWidgetDelegate; // free images list delegate
    delete treeWidgetDelegate; // free duplicates list delegate

    QCoreApplication::quit(); // quit program
}

/// Save and load

void MainWindow::SaveDirBaseFile() // write current folder name in ini file
{
    cv::FileStorage fs(basedirinifile, cv::FileStorage::WRITE); // open dir ini file for writing
    fs << "BaseDir" << basedir; // write folder name
    fs.release(); // close file
}

void MainWindow::ChangeBaseDir(QString filename) // set base dir and file
{
    QFileInfo fi(filename); // get file info
    if ((fi.exists()) and (!fi.isFile())) { // is it just a folder ?
        basedir = filename.toUtf8().constData(); // set default folder to this value
        basefile =""; // no filename
        SaveDirBaseFile(); // Save current path to ini file

        return; // exit
    }

    basefile = filename.toUtf8().constData(); // base file name and folder are used after to save other files

    // Remove extension if present
    size_t period_idx = basefile.rfind('.');
    if (std::string::npos != period_idx)
        basefile.erase(period_idx);

    basedir = basefile; // base folder is the filenmae without extension
    size_t found = basefile.find_last_of("\\/"); // find last directory in it
    std::string separator = basefile.substr(found, 1); // copy path separator (Linux is not the same as Windows)
    basedir = basedir.substr(0, found) + separator; // define base path, also modify path separators
    basefile = basefile.substr(found + 1); // delete path in base file name

    SaveDirBaseFile(); // Save current path to ini file
}

/// Images list

// Loading images : 2 methods = OpenCV or Qt

QPixmap MainWindow::LoadImagePix(const std::string &path, const std::string &engine) // return a Qt QPixmap from image file using different loading engines
{
    QPixmap result; // image to return

    if (engine == "opencv") { // loading image engine is OpenCV ?
        cv::Mat img = imread(path, cv::IMREAD_COLOR); // read image

        if (!img.empty()) // success ?
            return Mat2QPixmap(img); // return converted image from OpenCV to Qt format
    }
    else if (engine == "qt") { // loading image engine is Qt ?
        return QPixmap(QString::fromStdString(path)); // read image and return it
    }

    return QPixmap(); // only used if there was an error : return an empty image
}

cv::Mat MainWindow::LoadImageMat(const std::string &path, const std::string &engine) // return an OpenCV Mat from image file using different loading engines
{
    cv::Mat result; // comments are almost the same as LoadImagePix

    if (engine == "opencv") {
        cv::Mat img = imread(path, cv::IMREAD_COLOR);

        if (!img.empty())
            return img;
    }
    else if (engine == "qt") {
        QPixmap img = QPixmap(QString::fromStdString(path));

        if (!img.isNull())
            return QPixmap2Mat(img);
    }

    return cv::Mat();
}

void MainWindow::on_button_add_images_clicked() // button pressed -> add images to list
{
    if (images.empty())
        ui->label_no_images->setVisible(false); // don't show message "no images"

    QString dir = QFileDialog::getExistingDirectory(this, "Add images...", QString::fromStdString(basedir), QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks); // ask for folder

    if (dir.isNull() || dir.isEmpty()) // cancel ?
        return;

    ChangeBaseDir(dir); // save current path to ini file
    std::string folder = dir.toUtf8().constData(); // get folder as std::string

    PopulateImagesList(folder, ui->checkBox_recursive->isChecked()); // add all images from this folder to the images list
}

void MainWindow::on_button_images_clear_clicked() // button pressed -> clear images list (and also duplicates list)
{
    // variables
    images.clear(); // clear internal images list

    // images list
    ui->listWidget_image_list->clear(); // no images shown in images list
    ShowImagesListCount(); // show 0 files in images list

    // clear duplicates
    ClearDuplicates(); // clear duplicates list and enable gui items and clear some variables too
}

void MainWindow::ImagesListClick(QListWidgetItem *item) // click on item in images list -> toggle checked status
{
    if (item->checkState() == Qt::Checked) { // is the item checked ?
        item->setCheckState(Qt::Unchecked); // so uncheck it
    }
    else { // item is unchecked
        item->setCheckState(Qt::Checked); // so check it
    }
}

void MainWindow::ImagesListDoubleClick(QListWidgetItem *item) // double-click on item of images list -> show image in new window
{
    int imgNumber = item->data(Qt::UserRole).toInt(); // get image number in internal images list

    if ((images[imgNumber].deleted) or (images[imgNumber].error)) // image is not valid ?
        return; // exit

    QPixmap pix = LoadImagePix(images[imgNumber].fullPath, images[imgNumber].loadwith); // load the image file from disk
    int width = pix.width(); // image size
    int height = pix.height();
    if ((width > 1000) or (height > 1000)) // shown image is max 1000x1000px
        pix = pix.scaled(1000, 1000, Qt::KeepAspectRatio, Qt::SmoothTransformation); // so resize it if needed, keeping aspect ratio

    QSize screenSize = qApp->screens()[0]->size(); // get screen size

    QLabel *label_img = new QLabel (this); // new label widget
    label_img->setWindowFlags(Qt::Window | Qt::WindowStaysOnTopHint); // the new window stays on top, this way we can open several
    label_img->setWindowTitle(QString::number(width) + "x" + QString::number(height) + " - " + QString::fromStdString(images[imgNumber].basename)); // window title with image info
    label_img->setGeometry((screenSize.width() - pix.width()) / 2, (screenSize.height() - pix.height()) / 2, pix.width(), pix.height()); // set window size + position=centered
    label_img->setPixmap(pix); // add the image to widget
    label_img->show(); // show the new window

    ImagesListClick(item); // double-click problem : a single click is detected first so check/uncheck the same image in images list

    // question : is the created QLabel (with "new") properly freed from memory when the window is closed ? should I "delete" it ? valgrinder seems not to mind so probably NO
    // when the window is closed the widget surely "dies"
}

// Selection

void MainWindow::on_button_images_uncheck_all_clicked() // button pressed -> uncheck all images in images list
{
    for(int n = 0; n < ui->listWidget_image_list->count(); n++) { // parse all widget items
        QListWidgetItem* item = ui->listWidget_image_list->item(n); // current item
        item->setCheckState(Qt::Unchecked); // uncheck it
    }
}

void MainWindow::on_button_images_check_all_clicked() // button pressed -> check all images in images list
{
    for(int n = 0; n < ui->listWidget_image_list->count(); n++) { // almost same comment as on_button_images_uncheck_all_clicked()
        QListWidgetItem* item = ui->listWidget_image_list->item(n);
        item->setCheckState(Qt::Checked);
    }
}

void MainWindow::on_button_images_uncheck_all_selected_clicked() // button pressed -> uncheck selected images in images list
{
    for(int n = 0; n < ui->listWidget_image_list->count(); n++) { // parse all widget items
        QListWidgetItem* item = ui->listWidget_image_list->item(n); // current item
        if (item->isSelected()) // is it selected ?
            item->setCheckState(Qt::Unchecked); // yes -> uncheck it
    }
}

void MainWindow::on_button_images_check_all_selected_clicked() // button pressed -> check selected images in images list
{
    for(int n = 0; n < ui->listWidget_image_list->count(); n++) { // almost same comment as on_button_images_uncheck_all_selected_clicked()
        QListWidgetItem* item = ui->listWidget_image_list->item(n);
        if (item->isSelected())
            item->setCheckState(Qt::Checked);
    }
}

void MainWindow::on_button_images_check_invert_clicked() // button pressed -> invert check state in images list
{
    for(int n = 0; n < ui->listWidget_image_list->count(); n++) { // parse all widget items
        QListWidgetItem* item = ui->listWidget_image_list->item(n); // current item
        if (item->checkState() == Qt::Checked) // is the item checked ?
            item->setCheckState(Qt::Unchecked); // uncheck it
        else // item was not checked ?
            item->setCheckState(Qt::Checked); // check it
    }
}

void MainWindow::on_button_images_check_text_clicked() // button pressed -> find words and check images in images list
{
    Qt::CheckState checkedState = ui->checkBox_images->checkState(); // use the current reference check state near the text field

    for(int n = 0; n < ui->listWidget_image_list->count(); n++) { // parse all widget items
        QListWidgetItem* item = ui->listWidget_image_list->item(n); // current item
        if (item->text().contains(ui->lineEdit_images_check_text->text(), Qt::CaseInsensitive)) // is the searched text found in this item's text ?
            item->setCheckState(checkedState); // set its check state to reference check state
    }
}

// Remove and delete

void MainWindow::RemoveImageFromListImages(QListWidgetItem *item) // remove one item from image list by its pointer
{
    int numImageItem = ui->listWidget_image_list->row(item); // get the image index from QListWidget images list
    QListWidgetItem *imageItem = ui->listWidget_image_list->takeItem(numImageItem); // delete it from view
    delete imageItem; // free item memory
}

void MainWindow::on_button_images_hide_clicked() // button pressed -> hide checked images from list
{
    // progress - this operation can be long if the images list is huge
    ShowProgress(progress_prepare);
    ShowProgress(progress_run, "Removing images", 0, 1);
    ShowProgress(progress_update, 0);

    bool removed = false; // indicates if at least one image was removed
    for (int n = 0; n < ui->listWidget_image_list->count(); n++) { // parse all images in images list
        QListWidgetItem *item = ui->listWidget_image_list->item(n); // current item
        if (item->checkState() == Qt::Checked) { // is it checked ?
            int ref = item->data(Qt::UserRole).toInt(); // get image index in internal list
            images[ref].deleted = true; // set its flag to deleted
            RemoveImageFromListImages(item); // remove image from images list
            removed = true; // at least one image was removed
            n--; // one item left -> decrease current item index by 1
        }
    }

    if (removed) { // was at least one image removed from the list ?
        ShowImagesListCount();
        ClearDuplicates(); // clear duplicates list and enable gui items and clear some variables too
    }

    // GUI elements
    ShowProgress(progress_finished, "Images removed");
}

void MainWindow::on_button_images_hide_error_clicked() // button pressed -> hide images with errors from list
{
    // progress - this operation can be long if the images list is huge
    ShowProgress(progress_prepare);
    ShowProgress(progress_run, "Removing images with errors", 0, 1);
    ShowProgress(progress_update, 0);

    for (int n = 0; n < int(images.size()); n++) { // parse all internal images list
        if (images[n].error) { // is the image tagged as error ?
            RemoveImageFromListImages(images[n].imageItem); // remove it from view
            images[n].deleted = true; // tag image as deleted
        }
    }

    // GUI elements
    ShowProgress(progress_finished, "Images with errors removed");
    ShowImagesListCount();

    // no need to clear the duplicates list, these images are not parsed when comparing !
}

void MainWindow::on_button_images_delete_clicked() // butoon pressed -> delete images from list
{
    int sure = QMessageBox::question(this, "Delete files warning", "Are you sure you want to delete " + deleteFileMessage + " the image file(s)?", QMessageBox::Yes|QMessageBox::No); // delete, are you sure ?
    if (sure == QMessageBox::No) // don't delete files !
        return;

    // progress - this operation can be long if the images list is huge
    ShowProgress(progress_prepare);
    ShowProgress(progress_run, "Deleting images", 0, 1);
    ShowProgress(progress_update, 0);

    QString errors = ""; // get track of problems when deleting images

    bool deleted = false; // indicate if at least one image file was deleted
    for (int n = 0; n < ui->listWidget_image_list->count(); n++) { // parse images list
        QListWidgetItem *item = ui->listWidget_image_list->item(n); // this item
        if (item->checkState() == Qt::Checked) { // is it checked ?
            int ref = item->data(Qt::UserRole).toInt(); // get internal image index

            if (!DeleteFile(images[ref].fullPath.c_str())) { // error deleting image file ?
                errors += QString::fromStdString(images[ref].fullPath) + "\n"; // add entry to errors log
            }
            else { // file image was deleted successfully : remove it from images list
                images[ref].deleted = true; // set image flag as deleted
                RemoveImageFromListImages(item); // hide it from view
                deleted = true; // at least one image was deleted
                n--; // one image less -> decrease current image index by 1
            }
        }
    }

    if (deleted) { // at least one image file was deleted ?
        ShowImagesListCount();
        ClearDuplicates(); // clear duplicates list and enable gui items and clear some variables too
    }

    // GUI elements
    ShowProgress(progress_finished, "Images deleted");

    if (errors != "") { // errors were found ?
        QMessageBox::warning(this, "Errors deleting image files", "Some images files could not be deleted.\nHere is the list of images that failed:\n\n" + errors); // show them in message box
    }
}

//// Options

void MainWindow::on_spinBox_thumbnails_size_valueChanged(int size) // change thumbnails size
    // options are not available if images were loaded so no need to recompute anything else than gui elements
{
    thumbnailsSize = size; // new thumbnails size

    ui->listWidget_image_list->setIconSize(QSize(thumbnailsSize, thumbnailsSize)); // set new thumbnail size in images list
    ui->listWidget_image_list->setGridSize(QSize(230 - 180 + thumbnailsSize, 250 - 180 + thumbnailsSize)); // and also change the grid size accordingly
    ui->treeWidget_duplicates->setIconSize(QSize(thumbnailsSize, thumbnailsSize - 30)); // set new thumbnail size in duplicates list
    ui->treeWidget_duplicates->setColumnWidth(1, thumbnailsSize + 20); // and also change the corresponding column size accordingly
}

void MainWindow::on_spinBox_reduced_size_valueChanged(int size) // change working image size
    // options are not available if images were loaded so no need to recompute anything else
{
    reducedSize = size;
}

void MainWindow::on_spinBox_nb_features_valueChanged(int nb) // change number of features to find for features detection and homography
    // options are not available if features or homography were computed so no need to recompute anything else
{
    nbFeatures = nb;
}

void MainWindow::on_doubleSpinBox_threshold_valueChanged(double nb) // change threshold
    // options are not available if images were loaded so no need to recompute anything else
{
    threshold = nb;
}

/// Duplicates

// clear

void MainWindow::ClearDuplicates() // clear duplicates list and associated GUI elements and variables + show duplicates count
{
    // variables
    pairs.clear(); // clear images pairs scores
    groups.clear(); // clear images groups list

    // duplicates list
    ui->treeWidget_duplicates->clear(); // no images shown in duplicates list

    // enable all options in options tab
    ui->frame_group_thumbnails_size->setDisabled(false);
    ui->frame_group_reduced_size->setDisabled(false);
    ui->frame_group_nb_features->setDisabled(false);

    // disable algorithms in ui
    for (int i = img_similarity_checksum; i < img_similarity_count; i++) {
        HideCombinedAlgorithm(static_cast<imageSimilarityAlgorithm>(i), true);
        SetCombinedActivated(static_cast<imageSimilarityAlgorithm>(i), false);
    }

    ShowDuplicatesListCount(); // show 0 files in duplicates list
}

// Compute similarities

void MainWindow::on_button_compare_images_clicked() // compare images from images list
{
    CompareImages(); // the most important function in this program !
}

// Check

void MainWindow::DuplicatesListClick(QTreeWidgetItem *item, int column) // click anywhere (except the image) on duplicate item to check it
{
    if (!item->parent()) // black items are also clickable but are not real items
        return; // exit

    if (column != 1) { // you have to click anywhere BUT the image to check an item
        int state = item->checkState(0); // get item's check state
        if (state == Qt::Checked) { // is it checked ?
            item->setCheckState(0, Qt::Unchecked); // uncheck it !
            for (int i = 3; i < 6; i++) // change item colors to indicate it is unchecked (only on right side)
                item->setBackground(i, QColor(148, 148, 148)); // gray
        }
        else { // item was NOT checked
            item->setCheckState(0, Qt::Checked); // check it
            for (int i = 3; i < 6; i++) // change item colors to indicate it is checked (only on right side)
                item->setBackground(i, QColor(255, 153, 153)); // pale red
        }
    }
}

void MainWindow::on_button_duplicates_uncheck_all_clicked() // button pressed -> uncheck all duplicates in duplicates list
{
    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) { // parse all top duplicates items (groups)
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i); // current top item
        for (int j = 0; j < topItem->childCount(); j++) { // parse its children
            QTreeWidgetItem *item = topItem->child(j); // current child
            item->setCheckState(0, Qt::Unchecked); // uncheck it
            for (int k = 3; k < 7; k++) // change item colors to indicate it is unchecked (only on right side)
                item->setBackground(k, QColor(148, 148, 148)); // gray
        }
    }
}

void MainWindow::on_button_duplicates_check_all_clicked() // button pressed -> check all duplicates in duplicates list
{
    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) { // almost same comments as on_button_duplicates_uncheck_all_clicked()
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i);
        for (int j = 0; j < topItem->childCount(); j++) {
            QTreeWidgetItem *item = topItem->child(j);
            item->setCheckState(0, Qt::Checked); // check it
            for (int k = 3; k < 7; k++)
                item->setBackground(k, QColor(255, 153, 153)); // pale red
        }
    }
}

void MainWindow::on_button_duplicates_check_invert_clicked() // button pressed -> invert all checks in duplicates list
{
    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) { // almost same comments as on_button_duplicates_uncheck_all_clicked()
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i);
        for (int j = 0; j < topItem->childCount(); j++) {
            QTreeWidgetItem *item = topItem->child(j);
            if (item->checkState(0) == Qt::Unchecked) { // if unchecked check it
                item->setCheckState(0, Qt::Checked);
                for (int k = 3; k < 7; k++)
                    item->setBackground(k, QColor(255, 153, 153)); // pale red
            }
            else {
                item->setCheckState(0, Qt::Unchecked); // if checked uncheck it
                for (int k = 3; k < 7; k++)
                    item->setBackground(k, QColor(148, 148, 148)); // gray
            }
        }
    }
}

void MainWindow::on_button_duplicates_check_first_clicked() // button pressed -> check first duplicate in each group in duplicates list
{
    Qt::CheckState checkedState = ui->checkBox_duplicates->checkState(); // get reference checked state

    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) { // almost same comments as on_button_duplicates_uncheck_all_clicked()
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i);
        QTreeWidgetItem *item = topItem->child(0); // only test first child
        if (item) {
            item->setCheckState(0, checkedState);
            if (checkedState == Qt::Checked) {
                for (int k = 3; k < 7; k++)
                    item->setBackground(k, QColor(255, 153, 153));
            }
            else {
                for (int k = 3; k < 7; k++)
                    item->setBackground(k, QColor(148, 148, 148));
            }
        }
    }
}

void MainWindow::on_button_duplicates_check_rest_clicked() // button pressed -> check all duplicates in each group but first one in duplicates list
{
    Qt::CheckState checkedState = ui->checkBox_duplicates->checkState(); // almost same comments as oon_button_duplicates_check_first_clicked()

    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) {
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i);
        for (int j = 1; j < topItem->childCount(); j++) { // parse all children BUT the first one
            QTreeWidgetItem *item = topItem->child(j);
            item->setCheckState(0, checkedState);
            if (checkedState == Qt::Checked) {
                for (int k = 3; k < 7; k++)
                    item->setBackground(k, QColor(255, 153, 153));
            }
            else {
                for (int k = 3; k < 7; k++)
                    item->setBackground(k, QColor(148, 148, 148));
            }
        }
    }
}

void MainWindow::on_button_duplicates_check_text_clicked() // button pressed -> find words and check duplicates in duplicates list
{
    Qt::CheckState checkedState = ui->checkBox_duplicates->checkState(); // get reference checked state

    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) { // parse all top items (groups)
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i); // current top item
        for (int j = 0; j < topItem->childCount(); j++) { // parse all its children
            QTreeWidgetItem *item = topItem->child(j); // current child
            int imageNumber = item->data(0, Qt::UserRole).toInt(); // get image index in internal images list
            if (QString::fromStdString(images[imageNumber].fullPath).contains(ui->lineEdit_duplicates_check_text->text(), Qt::CaseInsensitive)) { // does the full image path contains the searched words ?
                item->setCheckState(0, checkedState); // current item
                if (checkedState == Qt::Checked) { // as in other functions before, check or uncheck
                    for (int k = 3; k < 6; k++)
                        item->setBackground(k, QColor(255, 153, 153));
                }
                else {
                    for (int k = 3; k < 6; k++)
                        item->setBackground(k, QColor(148, 148, 148));
                }
            }
        }
    }
}

// Remove and delete

void MainWindow::on_button_duplicates_hide_clicked() // button pressed -> hide checked duplicates in duplicates list
{
    bool hidden = false; // indicate at least one item was hidden
    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) {
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i);
        for (int j = 0; j < topItem->childCount(); j++) {
            QTreeWidgetItem *item = topItem->child(j);
            if (item->checkState(0) == Qt::Checked) { // is current item checked ?
                delete item; // delete it from duplicates view
                j--; // one duplicate less -> decrease current item index by 1
                hidden = true; // at least one item was hidden
            }
        }
        if (topItem->childCount() <= 1) { // if a top item is empty or only contains 1 item
            delete topItem; // delete it
            i--; // one top item less -> decrease current top item index by 1
        }
    }

    if (hidden) // at least one item was hidden ?
        ShowDuplicatesListCount(); // show new duplicates count
}

void MainWindow::on_button_duplicates_delete_clicked() // button pressed -> remove checked duplicates in duplicates list and delete the image file
{
    int sure = QMessageBox::question(this, "Delete files warning", "Are you sure you want to delete " + deleteFileMessage + " the image file(s)?", QMessageBox::Yes|QMessageBox::No); // delete, are you sure ?
    if (sure == QMessageBox::No) // don't delete files !
        return;

    // progress - operation can be long if duplicates list is huge
    ShowProgress(progress_prepare);
    ShowProgress(progress_run, "Deleting images files", 0, 1);
    ShowProgress(progress_update, 0);

    QString errors = ""; // error log
    bool deleted = false; // indicate that at least one duplicate image file was deleted
    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) { // parse all top items (groups)
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i); // current top item
        for (int j = 0; j < topItem->childCount(); j++) { // parse all its children
            QTreeWidgetItem *item = topItem->child(j); // current child
            if (item->checkState(0) == Qt::Checked) { // is it checked ?
                int imgNumber = item->data(0, Qt::UserRole).toInt(); // get internal image list index

                if (!DeleteFile(images[imgNumber].fullPath.c_str())) { // error deleting image file ?
                    errors += QString::fromStdString(images[imgNumber].fullPath) + "\n"; // add entry to errors log
                }
                else { // file image was deleted successfully : remove it from all lists
                    images[imgNumber].deleted = true;
                    delete item; // remove image from duplicates list
                    RemoveImageFromListImages(images[imgNumber].imageItem); // remove image from displayed images list
                    j--; // one duplicate less -> decrease current item index by 1
                    deleted = true; // at least one duplicate image file was deleted
                }
            }
        }
        if (topItem->childCount() <= 1) { // if a top item is empty or only contains 1 item
            delete topItem; // delete it
            i--; // one top item less -> decrease current top item index by 1
        }
    }

    if (deleted) { // at least one duplicate image file was deleted ?
        ShowImagesListCount(); // show images and duplicates new count
        ShowDuplicatesListCount();
    }

    ShowProgress(progress_finished, "Images removed");

    if (errors != "") { // error log not empty ? There were errors !
        QMessageBox::warning(this, "Errors deleting image files", "Some images files could not be deleted.\nHere is the list of images that failed:\n\n" + errors); // show log in message box
    }
}

void MainWindow::on_button_duplicates_copy_clicked() // button pressed -> copy files of checked duplicates in duplicates list to another folder
{
    QString dir = QFileDialog::getExistingDirectory(this, "Add images...", QString::fromStdString(basedir), QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks); // get destination folder

    if (dir.isNull() || dir.isEmpty()) // cancel ?
        return;

    dir += "/"; // add a folder separator to folder : this will be the base for all copied files
    std::string folder = dir.toUtf8().constData(); // convert it to std::string

    // progress - it could take some time if the duplicates list is huge
    ShowProgress(progress_prepare);
    ShowProgress(progress_run, "Copying image files", 0, 1);
    ShowProgress(progress_update, 0);

    QString errors = ""; // error log
    bool copyConfirm = false; // indicator for user's choice to confirm file replacement
    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) { // parse all top items (groups)
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i); // current top item
        for (int j = 0; j < topItem->childCount(); j++) { // parse all its children
            bool confirmThisTime = copyConfirm; // indicator for user's choice to confirm file replacement - ONCE - if "replace all" was chosen by user, it is always confirmed for this time...
            QTreeWidgetItem *item = topItem->child(j); // current child item
            if (item->checkState(0) == Qt::Checked) { // is it checked ? then we have to copy it
                int imgNumber = item->data(0, Qt::UserRole).toInt(); // get internal image index

                std::string newFilename = folder + images[imgNumber].basename; // destination full file path
                std::ifstream inFile(newFilename); // does it already exist ?
                if (inFile.good()) { // that means file already exists
                    if (!confirmThisTime) { // confirm replacement indicator ONCE not set ?
                        int confirm = QMessageBox::question(this, "Copying file image...",
                                                            "The file already exists.\nAre you sure you want to copy the file image?\n" + QString::fromStdString(images[imgNumber].fullPath),
                                                            QMessageBox::Yes|QMessageBox::No|QMessageBox::YesToAll); // copy, are you sure ?
                        if (confirm == QMessageBox::YesToAll) { // user confirmed replacement for all files ?
                            copyConfirm = true; // set indicator to copy file
                        }
                        else if (confirm == QMessageBox::Yes) { // user confirmed replacement for this file ?
                            confirmThisTime = true; // yes for ONCE
                        }
                    }
                }
                else { // file didn't exist in destination folder
                    confirmThisTime = true; // so copy it
                }

                if ((copyConfirm) or (confirmThisTime)) { // confirmation to copy the file ?
                    if (!std::filesystem::copy_file(images[imgNumber].fullPath.c_str(), newFilename.c_str(), std::filesystem::copy_options::overwrite_existing)) { // error copying image file ?
                        errors += QString::fromStdString(images[imgNumber].fullPath) + "\n"; // add entry to errors log
                    }
                }
            }
        }
    }

    ShowProgress(progress_finished, "Image files copied");

    if (errors != "") { // errors were found ?
        QMessageBox::warning(this, "Errors copying image files", "Some images files could not be copied.\nHere is the list of images that failed:\n\n" + errors); // show log in message box
    }
}

void MainWindow::on_button_duplicates_move_clicked() // button pressed -> move files of checked duplicates in duplicates list to another folder
{
    // all comments are almost the same as on_button_duplicates_copy_clicked(), except that files are MOVED

    QString dir = QFileDialog::getExistingDirectory(this, "Add images...", QString::fromStdString(basedir), QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);

    if (dir.isNull() || dir.isEmpty()) // cancel ?
        return;

    dir += "/";
    std::string folder = dir.toUtf8().constData();

    // progress
    ShowProgress(progress_prepare);
    ShowProgress(progress_run, "Moving image files", 0, 1);
    ShowProgress(progress_update, 0);

    QString errors = "";
    bool moveConfirm = false;
    bool moved = false; // indicate that at least one file was moved
    for (int i = 0; i < ui->treeWidget_duplicates->topLevelItemCount(); i++) {
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(i);
        for (int j = 0; j < topItem->childCount(); j++) {
            bool confirmThisTime = moveConfirm;
            QTreeWidgetItem *item = topItem->child(j);
            if (item->checkState(0) == Qt::Checked) {
                int imgNumber = item->data(0, Qt::UserRole).toInt();

                std::string newFilename = folder + images[imgNumber].basename;
                std::ifstream inFile(newFilename);
                if (inFile.good()) { // that means file already exists
                    if (!confirmThisTime) {
                        int confirm = QMessageBox::question(this, "Moving file image...", "Are you sure you want to move the file image?\n" + QString::fromStdString(images[imgNumber].fullPath), QMessageBox::Yes|QMessageBox::No|QMessageBox::YesToAll); // move, are you sure ?
                        if (confirm == QMessageBox::YesToAll) {
                            moveConfirm = true;
                        }
                        else if (confirm == QMessageBox::Yes) {
                            confirmThisTime = true;
                        }
                    }
                }
                else {
                    confirmThisTime = true;
                }

                if ((moveConfirm) or (confirmThisTime)) {
                    if (std::rename(images[imgNumber].fullPath.c_str(), newFilename.c_str()) != 0) { // error moving image file ?
                        errors += QString::fromStdString(images[imgNumber].fullPath) + "\n"; // add entry to errors log
                    }
                    else { // file image was moved successfully : update image data (internal and images list and duplicates)
                        moved = true; // at least one file was moved
                        images[imgNumber].fullPath = newFilename; // change full path in internal images list
                        images[imgNumber].folder = folder; // change folder in internal images list
                        item->setText(6, QString::fromStdString(images[imgNumber].folder)); // change folder in duplicates list
                        item->setToolTip(6, QString::fromStdString(images[imgNumber].fullPath)); // change tooltip in duplicates list
                        images[imgNumber].imageItem->setToolTip(QString::fromStdString(newFilename)); // change tooltip in images list

                        for (int n = 0; n < int(images.size()); n++) { // check for duplicates in internal images list with new filename
                            if ((n != imgNumber) and (images[n].fullPath == newFilename)) { // don't test current image itself ! is the filename the same ?
                                images[n].deleted = true; // delete it, it's a duplicate !
                                RemoveImageFromListImages(images[n].imageItem); // remove image from displayed images list
                                QTreeWidgetItem *top = images[n].duplicateItem->parent(); // get this duplicate's group in duplicates list
                                if (images[n].duplicateItem) // this duplicate was listed ?
                                    delete images[n].duplicateItem; // delete it from the list
                                if ((top) and (top->childCount() <= 1)) // is the parent valid ? is the group still valid ?
                                    delete top; // delete it
                                break; // duplicate found, no need to test the other children
                            }
                        }
                    }
                }
            }
        }
    }

    if (moved) { // at least one file was moved
        ShowImagesListCount();
        ShowDuplicatesListCount();
    }

    ShowProgress(progress_finished, "Image files moved");

    if (errors != "") {
        QMessageBox::warning(this, "Errors moving image files", "Some images files could not be moved.\nHere is the list of images that failed:\n\n" + errors);
    }
}

// Loading

void MainWindow::DuplicatesListDoubleClick(QTreeWidgetItem *item, int column) // double-click duplicate image to open it in a new window
{
    int imgNumber = item->data(0, Qt::UserRole).toInt(); // get index in internal images list

    if ((images[imgNumber].deleted) or (images[imgNumber].error)) // image can't be shown ?
        return; // exit

    if (column == 1) { // only image column can do this
        QPixmap pix = LoadImagePix(images[imgNumber].fullPath, images[imgNumber].loadwith); // load this image file
        int width = pix.width(); // size
        int height = pix.height();
        if ((width > 1000) or (height > 1000)) // shown image more than 1000x1000px ?
            pix = pix.scaled(1000, 1000, Qt::KeepAspectRatio, Qt::SmoothTransformation); // resize it to 1000x1000px

        QSize screenSize = qApp->screens()[0]->size(); // get current screen size

        QLabel *label_img = new QLabel (this); // as seen before in other functions, show image in a stay-on-top window
        //label_img->setWindowModality(Qt::WindowModal);
        label_img->setWindowFlags(Qt::Window | Qt::WindowStaysOnTopHint);
        label_img->setWindowTitle(QString::number(width) + "x" + QString::number(height) + " - " + QString::fromStdString(images[imgNumber].basename));
        label_img->setGeometry((screenSize.width() - pix.width()) / 2, (screenSize.height() - pix.height()) / 2, pix.width(), pix.height());
        label_img->setPixmap(pix);
        label_img->show();
    }

    // question : is the created QLabel (with "new") properly freed from memory when the window is closed ? should I "delete" it ? valgrinder seems not to mind so probably NO
    // when the window is closed the widget surely "dies"
}

// save results

void MainWindow::on_button_duplicates_save_results_clicked() // button to save results - for debugging purpose only - deactivate it for final version
{
    SaveResults();
}


/////////////////// Core functions //////////////////////

/// General

QColor MainWindow::ImageTypeColor(const std::string &extension) // returns image file type color
{
    if (extension == "jpeg")
        return QColor(255,255,204); // pale yellow
    if (extension == "tiff")
        return QColor(224,255,224); // pale green
    if (extension == "png")
        return QColor(224,255,255); // pale cyan
    if (extension == "webp")
        return QColor(255,224,224); // pale red

    return QColor(204,204,255); // other formats : violet
}

/// Images list

// Manage image list

void MainWindow::PopulateImagesList(const std::string &folder, const bool &recursive) // parse a directory and add images
{
    // image extensions to search
    QStringList fileExtension;
    // PNG OK
    QStringList() << "*.png";
    // JPEG and JPEG2000 OK
    QStringList() << "*.jpg";
    QStringList() << "*.jpeg";
    QStringList() << "*.jp2";
    QStringList() << "*.jpe";
    // TIFF OK
    QStringList() << "*.tif";
    QStringList() << "*.tiff";
    // WebP OK
    QStringList() << "*.webp";
    // Microsoft OK
    QStringList() << "*.bmp";
    QStringList() << "*.dib";
    // Portable Image Format OK
    QStringList() << "*.pbm";
    QStringList() << "*.pgm";
    QStringList() << "*.ppm";
    QStringList() << "*.pxm";
    QStringList() << "*.pnm";
    //// PFM !OK
    QStringList() << "*.pfm";
    // Sun raster OK
    QStringList() << "*.sr";
    QStringList() << "*.ras";
    //// OpenEXR
    QStringList() << "*.exr";
    // Radiance HDR OK
    QStringList() << "*.hdr";
    QStringList() << "*.pic";
    //// NOT supported by opencv but Qt
    //// X11 !OK
    QStringList() << "*.xbm";
    QStringList() << "*.xpm";
    // GIF OK
    QStringList() << "*.gif";
    // TGA OK
    QStringList() << "*.tga";
    // WBMP OK
    QStringList() << "*.wbmp";
    //// HEIC !OK
    QStringList() << "*.heic";
    //// not supported at all ?
    QStringList() << "*.heif";
    QStringList() << "*.avif";

    std::vector<QString> list; // to store dir results
    list.reserve(50000); // reserve memory for lists
    images.reserve(50000);

    QDirIterator::IteratorFlags flags; // flags for files search
    if (recursive) // recursive search ?
        flags = QDirIterator::Subdirectories | QDirIterator::FollowSymlinks; // set recursive flag
    else
        flags = QDirIterator::FollowSymlinks; // follow symlinks

    QDirIterator dir(QString::fromStdString(folder), fileExtension, QDir::Files, flags); // get files list
    while (dir.hasNext()) { // is there at least a file in list ?
        QFile f(dir.next()); // next item
        list.push_back(dir.filePath()); // store value
    }

    if (list.size() == 0) { // is the list empty ?
        QMessageBox::warning(this, "No image to add", "There was no image to add to the list.\nMaybe try the 'Recursive' option?"); // show error in message box
        return; // exit
    }

    #pragma omp parallel
    {
        #pragma omp for
        for (int n = 0; n < int(list.size()); n++) { // parse found files list
            struct_image_info img; // new image item

            img.newImage = true; // indicate it is a new image
            img.fullPath = list[n].toUtf8().constData(); // then fill all required fields
            img.basename = stringutils::GetFilenameFromFullPath(img.fullPath);
            img.folder = stringutils::GetFolderFromFullPath(img.fullPath);
            img.extension = stringutils::GetFilenameExtension(img.basename);
            img.deleted = false;

            // image file type will tell which engine (OpenCV or Qt) will be used to load it
            std::string ext = stringutils::ToLower(img.extension);
            if ((ext == "jpg") or (ext == "jpeg") or (ext == "jp2") or (ext == "jpe")) {
                img.type = "jpeg";
                img.loadwith = "opencv";
            }
            else if ((ext == "tif") or (ext == "tiff")) {
                img.type = "tiff";
                img.loadwith = "opencv";
            }
            else if (ext == "png") {
                img.type = "png";
                img.loadwith = "opencv";
            }
            else if (ext == "webp") {
                img.type = "webp";
                img.loadwith = "opencv";
            }
            else {
                img.type = "other";

                if ((ext == "bmp") or (ext == "dib"))
                    img.loadwith = "opencv";
                else if ((ext == "pbm") or (ext == "pgm") or (ext == "ppm") or (ext == "pxm") or (ext == "pnm"))
                    img.loadwith = "opencv";
                else if ((ext == "sr") or (ext == "ras"))
                    img.loadwith = "opencv";
                else if (ext == "exr")
                    img.loadwith = "opencv";
                else if ((ext == "hdr") or (ext == "pic"))
                    img.loadwith = "opencv";
                else if ((ext == "heic") or (ext == "heif"))
                    img.loadwith = "qt";
                else if (ext == "mng")
                    img.loadwith = "qt";
                else if (ext == "tga")
                    img.loadwith = "qt";
                else if (ext == "wbmp")
                    img.loadwith = "qt";
                else if (ext == "gif")
                    img.loadwith = "qt";
            }

            img.duplicates.reserve(50); // why 50 ? is it enough ?

            #pragma omp critical
            images.push_back(img); // add this image to the internal images list
        }
    }

    CleanImagesList(); // clean the images list (look for example for duplicates)
    ComputeImagesListInfo(); // compute all the other required images info

    ShowImagesList(); // show the images list in gui
}

void MainWindow::CleanImagesList() // delete duplicates and some marked images in images list
{
    // sort images list by path, ascending
    std::sort(images.begin(), images.end(),
              [](const struct_image_info& a, const struct_image_info& b) {
                    return (a.fullPath < b.fullPath);
              });

    // images to remove
    for (int n = 0; n < int(images.size()); n++) { // parse images list except last one
        if (images[n].deleted) { // image marked for deletion ?
            images.erase(images.begin() + n); // delete it
            n--; // index minus 1 because one image was deleted
        }
    }

    // delete duplicates in internal images list (NOT shown images in UI !)
    for (int n = 0; n < int(images.size()) - 1; n++) { // parse images list except last one
        if (images[n].fullPath == images[n + 1].fullPath) { // duplicate ?
            if (images[n].newImage) // is the current one a new image ?
                images.erase(images.begin() + n); // delete it
            else // current image NOT a new image -> delete the other one
                images.erase(images.begin() + n + 1); // delete it
            n--; // index minus 1 because one image was deleted
        }
    }
}

void MainWindow::ComputeImagesListInfo() // compute all other required info in images list
{
    // progress - this operation could be long if the image list is huge
    ShowProgress(progress_prepare);
    ShowProgress(progress_run, "Creating thumbnails", 0, int(images.size()));
    ShowProgress(progress_update, "Creating thumbnails", 0);

    int progress = 0;
    int count = 0;
    #pragma omp parallel
    {
        #pragma omp for
        for (int n = 0; n < int(images.size()); n++) { // parse images list
            if ((!stop) and (images[n].newImage)) { // is this a new image ?
                cv::Mat pix = LoadImageMat(images[n].fullPath, images[n].loadwith); // load the current image file

                if (pix.empty()) { // error reading image file ?
                    images[n].error = true; // marks image as not readable
                    images[n].deleted = false; // mark image as not available
                    images[n].width = 0; // image width
                    images[n].height = 0; // image height
                    images[n].imageSize = 0; // image size = width x height
                    images[n].icon = QPixmap(":/icons/image-error.png"); // show an error icon
                    images[n].newImage = false; // not a new image anymore
                    images[n].used = false; // won't be used anyway...
                }
                else { // no error -> continue
                    // image info
                    images[n].width = pix.cols; // get image width
                    images[n].height = pix.rows; // get image height
                    images[n].imageSize = images[n].width * images[n].height; // size = width x height
                    pix = QualityResizeImageAspectRatio(pix, cv::Size(reducedSize, reducedSize)); // resize image to working image size (see options tab)
                    // icon
                    cv::Mat icon = cv::Mat(thumbnailsSize, thumbnailsSize - 1, CV_8UC3); // size - 1 in vertical for display reasons (line under item in duplicates list)
                    icon = cv::Vec3b(148, 148, 148); // fill the icon image with gray
                    cv::Mat reduced = QualityResizeImageAspectRatio(pix, cv::Size(thumbnailsSize, thumbnailsSize)); // image icon
                    PasteImageColor(icon, reduced, (thumbnailsSize - reduced.cols) / 2, (thumbnailsSize - reduced.rows) / 2, false); // paste it upon the gray block
                    //cv::rectangle(icon, cv::Point(0, 0), cv::Point(icon.cols - 1, icon.rows - 1), cv::Vec3b(0, 0, 0), 1, cv::LINE_8);
                    //cv::line(icon, cv::Point(0, 0), cv::Point(icon.cols - 1, 0), cv::Vec3b(0, 0, 0), 1, cv::LINE_8);
                    cv::line(icon, cv::Point(0, 0), cv::Point(0, icon.rows - 1), cv::Vec3b(0, 0, 0), 1, cv::LINE_8); // draw vertical lines on left and right of the icon
                    cv::line(icon, cv::Point(icon.cols - 1, 0), cv::Point(icon.cols - 1, icon.rows - 1), cv::Vec3b(0, 0, 0), 1, cv::LINE_8);
                    images[n].icon = Mat2QPixmap(icon); // convert cv::Mat to QPixmap, store it in image item
                    // cached reduced image
                    images[n].imageReduced = pix; // reduced color image, store it too
                    // equalize histogram of reduced image
                    cv::Mat ycrcb; // will do it in YCrCb color space
                    cv::cvtColor(images[n].imageReduced, ycrcb, cv::COLOR_BGR2YCrCb); // convert image to color space
                    std::vector<cv::Mat> channels;
                    cv::split(ycrcb, channels); // split its channels
                    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(); // we will equalize with CLAHE algorithm
                    clahe->setClipLimit(4); // CLAHE options
                    clahe->apply(channels[0], channels[0]); // apply CLAHE to luminosity channel of image
                    //cv::equalizeHist(channels[0], channels[0]);
                    cv::merge(channels, ycrcb); // re-merge channels
                    cv::cvtColor(ycrcb, images[n].imageReduced, cv::COLOR_YCrCb2BGR); // convert back image from color space, store it
                    // gray reduced image
                    cv::cvtColor(images[n].imageReduced, images[n].imageReducedGray, cv::COLOR_BGR2GRAY); // convert reduced image to gray, store it

                    // flags
                    images[n].newImage = false; // not a new image anymore
                    images[n].deleted = false; // not deleted either
                    images[n].error = false; // and finally not an error !
                    images[n].used = false; // will be reset anyway
                }
            }

            progress++; // one more image done
            count++; // intermediate count
            if (omp_get_thread_num() == 0) { // is it the main parallel CPU thread ?
                if (count > 50) { // if count is enough
                    ShowProgress(progress_update, "", progress); // update progress in GUI
                    count = 0; // reset intermediate count
                }
            }
            if (stop) {
                n = images.size() - 1;
            }
        }
    }

    if (stop) {
        images.clear();
        ShowProgress(progress_finished, "Thumbnails creation canceled");
    }
    else {
        ShowProgress(progress_finished, "Thumbnails created");
    }
}

// Display

void MainWindow::ShowImagesList() // display images list from images information
{
    // (re)initialize widgets and lists
    ClearDuplicates(); // reset duplicates list

    // add images to widget
    for (int n = 0; n < int(images.size()); n++) { // parse images list
        if (!images[n].deleted) { // valid image ?
            QListWidgetItem *item = new QListWidgetItem; // create new image item for images list
            item->setIcon(images[n].icon); // icon
            item->setToolTip(QString::number(images[n].width) + "x" + QString::number(images[n].height) + " - " + QString::fromStdString(images[n].fullPath)); // tooltip
            item->setText(QString::fromStdString(images[n].basename)); // display image filename
            item->setTextAlignment(Qt::AlignHCenter | Qt::AlignVCenter); // alignment
            item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable); // item is enzbled, checkable and selectable
            item->setCheckState(Qt::Unchecked); // uncheck it (important, checkbox isn't shown if its state isn't set once)
            item->setData(Qt::UserRole, n); // insert index of images list in the widget, to retrieve it when needed
            item->setBackground(ImageTypeColor(images[n].type)); // color the background with image file type color

            ui->listWidget_image_list->addItem(item); // add image to widget
            images[n].imageItem = item; // keep track of item pointer in internal images list
        }
    }

    ShowImagesListCount(); // show number of images in GUI

    // options tab : no options are available but one : features number
    ui->frame_group_thumbnails_size->setDisabled(true);
    ui->frame_group_reduced_size->setDisabled(true);
}

void MainWindow::ShowImagesListCount() // show number of images in images list in GUI
{
    int count = 0; // count the images
    for (int n = 0; n < int(images.size()); n++) // parse images list
        if (!images[n].deleted) // valid image ?
            count++; // count it

    ui->lcdNumber_nb_images->display(count); // display count in LCD widget

    if (ui->listWidget_image_list->count() == 0) { // no images to display ?
        ui->label_no_images->setVisible(true); // show this information
    }
}

/// Duplicates

// Computing

cv::Point MainWindow::OrderedPair(const int &i, const int &j) // pairs must have im1 <= im2
{
    int x = j;
    int y = i;

    if (i < j) { // invert if i < j
        x = i;
        y = j;
    }

    return cv::Point(x, y);
}

int MainWindow::GetLevelFromScore(const imageSimilarityAlgorithm &similarityAlgorithm, const float &score) // get level of score from thresholds list - categories : 0 < dissimilar < different < similar < ∞ (exact)
{
    int level = 0; // start at minimum level
    while ((level < 3) and (thresholds[similarityAlgorithm][level] < score)) // is the score less important than current threshold for this similarity algorithm ?
        level++; // no enough, one more level

    return level; // should not be more than 3
}

float MainWindow::CombinedScore(const int &i, const int &j) // get combined score from previous tests
{
    if ((images[i].error) or (images[i].deleted) or (images[j].error) or (images[j].deleted)) // one of the image is invalid ?
        return 0; // exit

    auto pairScore = pairs.find(OrderedPair(i, j)); // find the score for this image pair

    if (pairScore == pairs.end()) // pair does not exist
        return 0; // exit

    int count = 0; // how many times the sum has been updated
    float sum = 0; // total weighted score
    for (int n = img_similarity_checksum; n < img_similarity_count; n++) { // parse all algorithms
        bool activated = false; // indicate if this score should be processed - each one is activated or not in the options tab
        switch (n) { // which algorithm ?
            case img_similarity_checksum:  activated = ui->checkBox_combined_checksum->checkState();break;
            case img_similarity_pHash:  activated = ui->checkBox_combined_phash->checkState();break;
            case img_similarity_dHash:  activated = ui->checkBox_combined_dhash->checkState();break;
            case img_similarity_idHash:  activated = ui->checkBox_combined_idhash->checkState();break;
            case img_similarity_block_mean:  activated = ui->checkBox_combined_blockmean->checkState();break;
            case img_similarity_marr_hildreth:  activated = ui->checkBox_combined_marrhildreth->checkState();break;
            case img_similarity_radial_variance:  activated = ui->checkBox_combined_radialvariance->checkState();break;
            //case img_similarity_color_moments:  activated = ui->checkBox_combined_colormoments->checkState();break;
            case img_similarity_dominant_colors:  activated = ui->checkBox_combined_dominantcolors->checkState();break;
            case img_similarity_features:  activated = ui->checkBox_combined_features->checkState();break;
            case img_similarity_homography:  activated = ui->checkBox_combined_homography->checkState();break;
            case img_similarity_dnn_classify:  activated = ui->checkBox_combined_classify->checkState();break;
        }

        if ((activated) and (pairScore->second.score[n] != -1)) { // if the score is to be processed and it is not invalid
            count ++; // count one more
            sum += pairScore->second.score[n] * float(GetLevelFromScore(static_cast<imageSimilarityAlgorithm>(n), pairScore->second.score[n])); // update sum with weight from thresholds
        }
    }

    return sum / (float(count) * 3.0f); // final result is the sum of scores divided by 3 times (4 - 1) levels and the count -> percentage
}

bool MainWindow::ImagesAreDuplicates(const int &i, const int &j, const imageSimilarityAlgorithm &similarityAlgorithm, const float &threshold, float &similarity) // compare a pair of images
{
    //// image orientation test

    switch (similarityAlgorithm) { // some algorithms won't work if images are not oriented the same way
        case img_similarity_checksum:
        case img_similarity_pHash:
        case img_similarity_dHash:
        case img_similarity_idHash:
        case img_similarity_block_mean:
        case img_similarity_marr_hildreth:
        case img_similarity_radial_variance: {
            // find orientation for image I
            float ratioI = float(images[i].width) / float(images[i].height); // compute ratio
            bool imIPortrait = true; // indicator if the image is portrait-oriented
            /*if (ratioI < 0.95f)
                imIPortrait = true; // it is portrait !
            else */if (ratioI > 1.05f) // test if image is landscape-oriented - some algorithms use a heavy resizing of images so 5% of difference is not a difference
                imIPortrait = false;
            // do the same for image J
            float ratioJ = float(images[j].width) / float(images[j].height);
            bool imJPortrait = true;
            if (ratioJ > 1.05f)
                imJPortrait = false;
            // final result
            if (imIPortrait != imJPortrait) { // orientation is not the same ?
                similarity = 0; // no similarity
                return false; // exit with value false
            }
            break;
        }
    }

    bool duplicate = false; // duplicate is false until proven true !

    //// create image hash/features/etc - keep result in cache

    if (similarityAlgorithm == img_similarity_dominant_colors) { // dominant colors : color image
        #pragma omp critical // because std::vectors will be used
        {
            if (images[i].dominantColors.empty()) { // for image I - if palette is not already computed
                cv::Mat reduced = ResizeImageAspectRatio(images[i].imageReduced, cv::Size(64, 64)); // resize image to a tiny size
                reduced = ConvertImageRGBtoOKLAB(reduced); // convert it to OKLAB color space
                cv::Mat quantized;
                images[i].dominantColors = DominantColorsEigen(reduced, 8, quantized); // quantize it with Eigen method, keep the resulting palette
            }
            if (images[j].dominantColors.empty()) { // same for image J
                cv::Mat reduced = ResizeImageAspectRatio(images[j].imageReduced, cv::Size(64, 64));
                reduced = ConvertImageRGBtoOKLAB(reduced);
                cv::Mat quantized;
                images[j].dominantColors = DominantColorsEigen(reduced, 8, quantized);
            }
        }
    }
    else if (similarityAlgorithm == img_similarity_features) { // image features (keypoints and descriptors) - gray image
        #pragma omp critical // because std::vectors will be used
        {
            if (images[i].keypoints.empty()) // for image I - if keypoints were not already computed
                ComputeImageDescriptors(images[i].imageReducedGray, images[i].keypoints, images[i].descriptors, false, reducedSize, nbFeatures); // compute keypoints
            if (images[j].keypoints.empty()) // same for image J
                ComputeImageDescriptors(images[j].imageReducedGray, images[j].keypoints, images[j].descriptors, false, reducedSize, nbFeatures);
        }

    }
    else if (similarityAlgorithm == img_similarity_homography) { // homography - same comments than features : homography is just a supplementary step from features - gray image
        #pragma omp critical
        {
            if (images[i].keypoints.empty())
                ComputeImageDescriptors(images[i].imageReducedGray, images[i].keypoints, images[i].descriptors, false, reducedSize, nbFeatures);
            if (images[j].keypoints.empty())
                ComputeImageDescriptors(images[j].imageReducedGray, images[j].keypoints, images[j].descriptors, false, reducedSize, nbFeatures);
        }
    }
    else if (similarityAlgorithm == img_similarity_dnn_classify) { // DNN classification
        if (images[i].hashDNN.empty()) { // image I - if classes are not already computed
            // VGG-16 : size=224, mean=(123.68, 116.779, 103.939))
            // Inception-21k : size=224, mean=(117, 117, 117)
            images[i].hashDNN = DNNHash(images[i].imageReduced, dnn, 224, cv::Scalar(117, 117, 117), 16); // compute classes using Inception-21k model
        }
        if (images[j].hashDNN.empty()) { // same for image J
            images[j].hashDNN = DNNHash(images[j].imageReduced, dnn, 224, cv::Scalar(117, 117, 117), 16);
        }
    }
    /*else if (similarityAlgorithm == img_similarity_color_moments) {
        if (images[i].hashTmp.empty())
            images[i].hashTmp = ImageHash(images[i].imageReduced, similarityAlgorithm);
        if (images[j].hashTmp.empty())
            images[j].hashTmp = ImageHash(images[j].imageReduced, similarityAlgorithm);
    }*/
    else if (similarityAlgorithm != img_similarity_count) { // NOT combined scores
        if (similarityAlgorithm != img_similarity_checksum) { // all other algorithms but checksum : gray image
            if (images[i].hashTmp.empty()) // image I - if the hash is not already computed
                images[i].hashTmp = ImageHash(images[i].imageReducedGray, similarityAlgorithm); // get it from corresponding algorithm
            if (images[j].hashTmp.empty()) // same for image J
                images[j].hashTmp = ImageHash(images[j].imageReducedGray, similarityAlgorithm);
        }
        else { // checksum uses the original images
            if (images[i].hashTmp.empty()) { // image I - if the hash is not already computed
                cv::Mat image = LoadImageMat(images[i].fullPath, images[i].loadwith); // load original image
                images[i].hashTmp = ImageHash(image, similarityAlgorithm); // hash it with MD5
            }
            if (images[j].hashTmp.empty()) { // same for image J
                cv::Mat image = LoadImageMat(images[j].fullPath, images[j].loadwith);
                images[j].hashTmp = ImageHash(image, similarityAlgorithm); // MD5
            }
        }
    }

    //// get images I and J's comparison score
    //// the score is computed or in cache
    //// all scores are percentages, the highest (100%) the better !

    // keep the score in a pair
    cv::Point pairPoint = OrderedPair(i, j); // index for images I and J, index-ordered
    auto pair = pairs.find(pairPoint); // find the pair

    // get score
    float match = 0; // by default, the lowest score

    if (similarityAlgorithm == img_similarity_count) { // combined score, NOT in cache because algorithms are chosen in the options tab
        match = CombinedScore(i, j); // compute it from scores cache
    }
    else if ((pair != pairs.end()) and (pair->second.score[similarityAlgorithm] != -1)) { // pair x,y already exists and score exists ?
        match = pair->second.score[similarityAlgorithm]; // no need to recompute, the score already exists
    }
    else if (similarityAlgorithm == img_similarity_dominant_colors) { // dominant colors : compare palettes
        match = (1.0f - CompareImagesDominantColorsFromEigen(images[i].dominantColors, images[j].dominantColors)) * 100.0f;
    }
    else if (similarityAlgorithm == img_similarity_features) { // features : compare image descriptors
        #pragma omp critical // because std::vectors are created
        {
            // if (images[i].imageReducedGray.cols * images[i].imageReducedGray.rows <= images[j].imageReducedGray.cols * images[j].imageReducedGray.rows)
                match = CompareImagesDescriptors(images[i].descriptors, images[j].descriptors, 0.8f, nbFeatures);
            /*else
                match = CompareImagesDescriptors(images[j].descriptors, images[i].descriptors, 0.8f, nbFeatures);*/

            match *= 100.0f;
        }
    }
    else if (similarityAlgorithm == img_similarity_homography) { // homography : get homography 3x3 matrix first, then compare the projection applied to one image
        cv::Mat homography = cv::Mat();
        #pragma omp critical // because std::vectors are created
        {
            std::vector<cv::Point2f> goodPoints1, goodPoints2; // not really used here but "good matching points" are needed/computed anyway

            if (images[i].imageReducedGray.cols * images[i].imageReducedGray.rows <= images[j].imageReducedGray.cols * images[j].imageReducedGray.rows) // works better if 2nd image is bigger than the 1st
                homography = GetHomographyFromImagesFeatures(images[i].imageReducedGray, images[j].imageReducedGray,
                                                             images[i].keypoints, images[j].keypoints,
                                                             images[i].descriptors, images[j].descriptors,
                                                             goodPoints1, goodPoints2,
                                                             match,
                                                             false, reducedSize, false, 0.8f, nbFeatures); // get the 3x3 homography matrix
            else // 1st image is bigger
                homography = GetHomographyFromImagesFeatures(images[j].imageReducedGray, images[i].imageReducedGray,
                                                             images[j].keypoints, images[i].keypoints,
                                                             images[j].descriptors, images[i].descriptors,
                                                             goodPoints1, goodPoints2,
                                                             match,
                                                             false, reducedSize, false, 0.8f, nbFeatures);
            match *= 100.0f;
        }
    }
    else if (similarityAlgorithm == img_similarity_dnn_classify) { // DNN classification
        match = DNNCompare(images[i].hashDNN, images[j].hashDNN) * 100.0f; // compare the classes
    }
    else { // hash-type algorithm
        match = ImageHashCompare(images[i].hashTmp, images[j].hashTmp, similarityAlgorithm); // compare the hashes
    }

    //// final result : are images similar ?

    switch (similarityAlgorithm) { // which algorithm ?
        case img_similarity_checksum: { // checksum : same or diffrent, the only algorithm that is binary
            if (match >= threshold) { // threshold should be 100%
                if ((images[i].width == images[j].width) and (images[i].height == images[j].height)) { // check images sizes to eliminate checksum collisions (not perfect but should work at 99.999%)
                    duplicate = true; // same sizes -> images are duplicates !
                }
            }
            break;
        }
        default: { // all other matching scores
            if (match >= threshold) // is the score more than threshold ?
                duplicate = true; // images are duplicates !
        }
    }

    similarity = match; // returned values
    return duplicate;
}

std::string MainWindow::GetHashString(const int &imageNumber, const imageSimilarityAlgorithm &similarityAlgorithm) // get hash string from image hash
    // only used for saving scores, only works for hashes
{
    switch (similarityAlgorithm) {
        case img_similarity_checksum: {
            return HashChecksum2String(images[imageNumber].hashTmp);
        }
        case img_similarity_aHash: {
            return Hash8U2String(images[imageNumber].hashTmp);
        }
        case img_similarity_pHash: {
            return Hash8U2String(images[imageNumber].hashTmp);
        }
        case img_similarity_dHash: {
            return Hash8U2String(images[imageNumber].hashTmp);
        }
        case img_similarity_idHash: {
            return Hash8U2String(images[imageNumber].hashTmp);
        }
        case img_similarity_block_mean: {
            return Hash8U2String(images[imageNumber].hashTmp);
        }
        case img_similarity_marr_hildreth: {
            return Hash8U2String(images[imageNumber].hashTmp);
        }
        case img_similarity_radial_variance: {
            return Hash8U2String(images[imageNumber].hashTmp);
        }
    }

    return "";
}

void MainWindow::CompareImages() // compare images in images list
{
    //// image list empty ?
    if (images.empty()) {
        QMessageBox::warning(this, "No images to compute...", "There are no images to compute.\nPlease add images first...\n");
        return;
    }

    //// reset duplicates list in gui
    ui->treeWidget_duplicates->clear(); // clear duplicates list widget - it is NOT a complete clear, just the shown images -> don't use ClearDuplicates() !
    ShowDuplicatesListCount(); // count should be zero
    ui->label_no_duplicates->setVisible(false); // hide the "no images" sign... until proven right

    //// gui
    if ((similarityAlgorithm == img_similarity_features) or (similarityAlgorithm == img_similarity_homography)) // for features and homography
        ui->frame_group_nb_features->setDisabled(true); // hide nb of features option in options tab

    //// clean cached info in images list
    for (int n = 0; n < int(images.size()); n++) { // parse all images
        images[n].duplicates.clear(); // duplicates list for this image
        images[n].group = -1; // no group assigned
        images[n].used = -1; // not already used in a group
        images[n].hashTmp = cv::Mat(); // empty hash cache
    }
    groups.clear(); // no group defined

    //// progress
    int progress = 0; // overall progression
    int count = 0; // for gui refresh
    int countLimit = 4000;
    if (similarityAlgorithm == img_similarity_dominant_colors)
        countLimit = 40;
    else if (similarityAlgorithm == img_similarity_dnn_classify)
             countLimit = 10;
    int sum = images.size() * (images.size() - 1) / 2; // number of comparisons to perform = 1+2+3+4+... images - formula is n(n+1)/2
    ShowProgress(progress_prepare);
    ShowProgress(progress_run, "Comparing images", 0, sum);
    ShowProgress(progress_update, "", 0);

    //// DNN initialization
    if ((similarityAlgorithm == img_similarity_dnn_classify) and (dnn.empty())) { // if DNN algorithm and not already defined
        DNNPrepare(dnn, "models/Inception21k.caffemodel", "models/Inception21k-bn.prototxt"); // prepare DNN model Unception 21K
    }

    //// find duplicates in images list
    for (int i = 0; i < int(images.size()) - 1; i++) { // parse images list minus last one, first pass
        if ((!images[i].deleted) and (!images[i].error)) { // valid image ?
            #pragma omp parallel
            {
                #pragma omp for
                for (int j = i + 1; j < int(images.size()); j++) { // parse images list, second pass - all preceding images have already been tested
                    if ((!stop) and (!images[j].deleted) and (!images[j].error)) { // image J valid ?
                        float similarity = -1; // default similarity : score not possible (should be 0 to 100%)
                        bool duplicates = ImagesAreDuplicates(i, j, similarityAlgorithm, threshold, similarity); // check if images I and J are duplicates, get also the score

                        #pragma omp critical // because std::vectors will be used
                        {
                            if (duplicates) { // images are duplicates ?
                                images[i].duplicates.push_back(j); // add each image to the duplicates list of the other one
                                images[j].duplicates.push_back(i);
                            }

                            cv::Point imagePair = OrderedPair(i, j); // index for images I and J, index-ordered
                            auto pair = pairs.find(imagePair); // get their similarity score if it exists

                            if ((pairs.empty()) or (pair == pairs.end())) { // pair doesn't exist so create it
                                struct_scores pair; // new pair - scores should already be set to -1 in constructor
                                pair.score[similarityAlgorithm] = similarity; // save the score between images I and J for the current algorithm
                                pairs.insert(std::make_pair(imagePair, pair)); // insert the result in the ist
                            }
                            else { // pair I,J already exists
                                pair->second.score[similarityAlgorithm] = similarity; // save the score between images I and J for the current algorithm
                            }
                        }
                    }

                    progress++; // one more comparison done !
                    count++; // counter
                    if (count > countLimit) { // enough images were compared ? time to update the progress bar
                        if (omp_get_thread_num() == 0) { // only the 1st CPU thread can to this
                            ShowProgress(progress_update, "", progress); // update progress bar with new value
                            count = 0; // reset counter
                        }
                    }
                    if (stop) {
                        i = images.size() - 1;
                        j = images.size() - 1;
                    }
                }
            }

            //hashFile << images[i].fullPath << ";" << GetHashString(i, similarityAlgorithm) << std::endl;
        }
    }

    ////  now all images are compared.. or operation is canceled

    if (stop) {
        ShowProgress(progress_finished, "Images comparison canceled");
        ui->label_no_duplicates->setVisible(true);
    }
    else {
        ShowDuplicatesList(); // display the similarity check results

        ShowProgress(progress_finished, "Images compared"); // end the current progress (that hides the animated wainting icon and restores the mouse cursor

        HideCombinedAlgorithm(similarityAlgorithm, false); // reveal current algorithm in options tab
        SetCombinedActivated(similarityAlgorithm, true); // and activate its checkbox
    }
}

// Display

QTreeWidgetItem* MainWindow::GetDuplicateItem(const int &ref) // add dpulicate info to item in duplicates view
{
    QTreeWidgetItem *item = new QTreeWidgetItem(); // add an item in list

    // column 0 - checkbox and image index
    item->setText(0, "");
    item->setTextAlignment(0, Qt::AlignHCenter | Qt::AlignVCenter);
    item->setData(0, Qt::UserRole, ref); // internal data = image index in images list -> this way a selected row can be associated with the original image
    // column 1 - icon
    item->setIcon(1, QIcon(images[ref].icon)); // column 1
    // column 2 - image file extension/type
    item->setText(2, QString::fromStdString(stringutils::ToUpper(images[ref].type))); // image file type text
    item->setBackground(2, ImageTypeColor(images[ref].type)); // set background color according to image type
    item->setTextAlignment(2, Qt::AlignHCenter | Qt::AlignVCenter);
    // column 3 - size
    item->setText(3, QString::number(images[ref].width) + "x" + QString::number(images[ref].height));
    item->setTextAlignment(3, Qt::AlignHCenter | Qt::AlignVCenter);
    // column 4 - resolution
    item->setText(4, QString::number(float(images[ref].width * images[ref].height) / 1000000.0f, 'f', 2)); // megapixels
    item->setTextAlignment(4, Qt::AlignHCenter | Qt::AlignVCenter);
    // column 5 - filename
    item->setText(5, QString::fromStdString(images[ref].basename));
    item->setToolTip(5, QString::fromStdString(images[ref].basename));
    // column 6 - file path
    item->setText(6, QString::fromStdString(images[ref].folder));
    item->setToolTip(6, QString::fromStdString(images[ref].folder));
    // general : new item options
    item->setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    item->setCheckState(0, Qt::Unchecked); // necessary, if the initial value is not set, the checkbox won't appear
    item->setSelected(false); // don't select the current line
    images[ref].duplicateItem = item; // keep the pointer to the item in the internal images list

    return item; // return lovingly handcrafted item
}

int MainWindow::GetClosestNeighbourNotUsed(const int &imageNumber, const imageSimilarityAlgorithm &algo) // get closest imageNumber of a imageNumber NOT used
    // used for 1st pass of clustering process
{
    int neighbour = -1; // means no image neighbour found
    float scoreMax = threshold; // initialize highest score found to threshold value (minimum score to attain) for this algorithm

    for (int currentNeighbour = 0; currentNeighbour < int(images[imageNumber].duplicates.size()); currentNeighbour++) { // parse all duplicates ("neighbours") for current image
        int ref = images[imageNumber].duplicates[currentNeighbour]; // get neighbour index in internal images list
        if ((ref != imageNumber) and (!images[ref].error) and (!images[ref].deleted) and (!images[ref].used)) { // neighbour valid and NOT equal to the tested image AND it is not already used
            auto pairScore = pairs.find(OrderedPair(imageNumber, ref)); // find the score for this pair
            if (pairScore->second.score[algo] >= scoreMax) { // is the current score more than highest score found ?
                neighbour = ref; // this image neighbour looks good !
                scoreMax = pairScore->second.score[algo]; // update highest score found
            }
        }
    }

    return neighbour; // return the image neighbour with the highest score... or -1 if no neighbour was found
}

int MainWindow::GetClosestNeighbour(const int &imageNumber, const imageSimilarityAlgorithm &algo) // get closest imageNumber of a imageNumber
    // only difference with GetClosestNeighbourNotUsed() is that we find a neighbour without checking if it already used
    // used for 1st and 2nd pass of clustering process
    // almost same comments as the other function
{
    int neighbour = -1;
    float scoreMax = ui->doubleSpinBox_threshold->value();

    for (int n = 0; n < int(images[imageNumber].duplicates.size()); n++) {
        int ref = images[imageNumber].duplicates[n];
        if ((ref != imageNumber) and (!images[ref].error) and (!images[ref].deleted)) {
            auto pairScore = pairs.find(OrderedPair(imageNumber, ref));
            if (pairScore->second.score[algo] >= scoreMax) {
                neighbour = ref;
                scoreMax = pairScore->second.score[algo];
            }
        }
    }

    return neighbour;
}

int MainWindow::GetClosestNeighbourHasGroup(const int &imageNumber, const imageSimilarityAlgorithm &algo) // get closest imageNumber of a imageNumber with a group
    // only difference with GetClosestNeighbourNotUsed() and GetClosestNeighbourNotUsed() is that we find a neighbour already used in a group
    // used only for 2nd pass of clustering process
    // almost same comments as the other functions
{
    int neighbour = -1;
    float scoreMax = ui->doubleSpinBox_threshold->value();

    for (int n = 0; n < int(images[imageNumber].duplicates.size()); n++) {
        int ref = images[imageNumber].duplicates[n];
        if ((ref != imageNumber) and (!images[ref].error) and (!images[ref].deleted) and (!groups[ref].empty())) {
            auto pairScore = pairs.find(OrderedPair(imageNumber, ref));
            if (pairScore->second.score[algo] >= scoreMax) {
                neighbour = ref;
                scoreMax = pairScore->second.score[algo];
            }
        }
    }

    return neighbour;
}

float MainWindow::GetScore(const int &im1, const int &im2, const imageSimilarityAlgorithm &algo) // get algo distance from 2 images
{
    auto pairScore = pairs.find(OrderedPair(im1, im2)); // look for the the score

    if (pairScore != pairs.end()) // score found ?
        return pairScore->second.score[algo]; // return it for current algorithm

    return 0; // not found -> return lowest score (should NEVER happen !)
}

void MainWindow::AddImageToGroup(const int &group, const int &image) // add an image to a group
{
    groups[group].push_back(image); // add this image to the group
    images[group].used = true; // indicate that both the image and the "head" image with same number as the group are now used in a group
    images[image].used = true;
}

void MainWindow::AddImageToGroupCheckScores(const int &group, const int &image, const imageSimilarityAlgorithm &algo) // add an image to a group with minimum score
{
    // check all group members, if only one has a bad score with the tested image, don't add it
    // this naturally creates very "close" images groups
    int count = 0; // counter for sufficiently similar images
    for (int ref = 0; ref < int(groups[group].size()); ref++) { // parse all images in this group
        int neighbour = groups[group][ref]; // current image neighbour
        auto pairScore = pairs.find(OrderedPair(image, neighbour)); // find pair's score
        if (pairScore->second.score[algo] >= threshold) // the score between current tested images is sufficient (>= threshold)
            count++; // one more image close to the image to add
    }

    // also check "head" of the group
    auto pairScore = pairs.find(OrderedPair(image, group));
    if (pairScore->second.score[algo] >= threshold)
        count++;

    float percentage = float(count) / float(groups[group].size() + 1); // percentage of "good" matches ?
    if (percentage <= 0.5) // if less than half of the images in this group match sufficiently with the main tested image
        return; // don't add the image to the group

    groups[group].push_back(image); // add this image to the group
    images[group].used = true; // indicate that both the "head" image and the main tested image are now used
    images[image].used = true;
}

void MainWindow::ShowDuplicatesList() // show list of duplicate images
{
    //// clear clusetring data in images internal list
    for (int n = 0; n < int(images.size()); n++) { // parse images
        images[n].used = false; // NOT used in any group
        images[n].group = -1; // NO group defined
    }

    //// GUI
    ui->treeWidget_duplicates->clear(); // clear duplicates list widget
    ui->label_no_duplicates->setVisible(false); // hide the "no image to display" message... until proven wrong

    //// create groups empty skeleton from images list
    groups.clear(); // clear groups data
    groups.reserve(images.size()); // reserve space in memory for it
    for (int i = 0; i < int(images.size()); i++) { // parse all images
        std::vector<int> group; // create an empty group and add it
        groups.push_back(group);
    }

    //// add images to groups - 1st pass
    // group #n is the #n image's group, it can be empty

    for (int currentImage = 0; currentImage < int(images.size()); currentImage++) { // parse all images
        if ((!images[currentImage].deleted) and (!images[currentImage].error) and (!images[currentImage].used)) { // current image is valid and not used ?
            int neighbour = GetClosestNeighbourNotUsed(currentImage, similarityAlgorithm); // find its closest neighbour in its duplicates list
            int oldNeighbour = -1; // keep track of last neighbour found
            while (neighbour != -1) { // while there is a neighbour
                float scoreCurrent = GetScore(currentImage, neighbour, similarityAlgorithm); // get the score for the current pair of images
                int neighbourClosestNeighbour = GetClosestNeighbourNotUsed(neighbour, similarityAlgorithm); // find the neighbour of the neighbour, NOT used
                if (neighbourClosestNeighbour != -1) { // a neighbour of the neighbour found ?
                    if (neighbourClosestNeighbour == currentImage) { // if the neighbour of the neighbour is the tested image itself
                        AddImageToGroupCheckScores(currentImage, neighbour, similarityAlgorithm); // add it to tested image's group, it's the best match we can find !
                    }
                    else { // no neighbour of the neighbour found
                        float scoreNeighbour = GetScore(neighbour, neighbourClosestNeighbour, similarityAlgorithm); // get the score between the image and its closest neighbour

                        if (scoreCurrent >= scoreNeighbour) { // if the neighbour is sufficiently similar to the image
                            AddImageToGroupCheckScores(currentImage, neighbour, similarityAlgorithm); // add it to the tested image's group
                        }
                    }
                }

                oldNeighbour = neighbour; // keep the last neighbour tested
                neighbour = GetClosestNeighbourNotUsed(currentImage, similarityAlgorithm); // find another one
                if (neighbour == oldNeighbour) // not any new neighbour found ?
                    neighbour = -1; // set exit condition for the while loop
            }
        }
    }
    /*simplest culstering !
        for (int currentImage = 0; currentImage < int(images.size()); currentImage++) { // parse all images
        if ((!images[currentImage].deleted) and (!images[currentImage].error) and (!images[currentImage].used) and (!images[currentImage].duplicates.empty())) { // current image is not used
            for (int indexNeighbour = 0; indexNeighbour < int(images[currentImage].duplicates.size()); indexNeighbour++) {
                int neighbour = images[currentImage].duplicates[indexNeighbour];
                if (!images[neighbour].used) { // current image is not used
                    AddImageToGroup(currentImage, neighbour);
                    images[neighbour].used = true;
                }
            }
            images[currentImage].used = true;
        }
    }*/

    //// add images to groups - 2nd pass
    // check if an image was not used at all (it happens !), in this case add it to the closest neighbours's group

    for (int currentImage = 0; currentImage < int(images.size()); currentImage++) { // parse all images
        if ((!images[currentImage].error) and (!images[currentImage].deleted) and (!images[currentImage].duplicates.empty()) and (!images[currentImage].used)) { // image is valid, has duplicates but is not used yet
            int neighbour = GetClosestNeighbourHasGroup(currentImage, similarityAlgorithm); // find the closest neighbour already assigned to a group
            if (neighbour != -1) { // valid neighbour found ?
                AddImageToGroup(neighbour, currentImage); // add unused image to this neighbour's group
            }
            else { // no closest neighbour with a group was found
                neighbour = GetClosestNeighbour(currentImage, similarityAlgorithm); // check if a closest neighbour exists (without an assigned group ! that was tested before)

                if (neighbour != -1) { // valid neighbour found ?
                    int neighbourNeighbour = GetClosestNeighbourHasGroup(neighbour, similarityAlgorithm); // check if this neighbour has a neighbour with a group

                    if (neighbourNeighbour != -1) // valid neighbour of neighbour ?
                        AddImageToGroup(neighbourNeighbour, currentImage); // add current image to this "far" group
                }
            }

            if (!images[currentImage].used) { // image didn't make it but has similarity... problem !
                qDebug() << "No group : image " << currentImage << " - " << QString::fromStdString(images[currentImage].fullPath); // log this error (should not often happen)
            }
        }
    }

    //// display image duplicates in groups : create tree widget items from groups list

    for (int group = 0; group < int(groups.size()); group++) { // parse groups list - group number is also the image's number
        if (groups[group].size() > 0) { // this image's group contains members ?
            // create a blue ligne = top
            QTreeWidgetItem *top = new QTreeWidgetItem(ui->treeWidget_duplicates); // new item
            for (int n = 0; n < 7; n++) // color this line in light cyan
                top->setBackground(n, QColor(230, 255, 255));
            // first item of this group is the image itself
            QTreeWidgetItem *first = GetDuplicateItem(group); // create first child item
            top->addChild(first); // add it to top item
            images[group].group = group; // set image's group to new group number

            // settooltip to group bar ?

            // now add image's duplicates to current group
            for (int j = 0; j < int(groups[group].size()); j++) { // parse this group's members
                int ref = groups[group][j]; // current image number of this group
                QTreeWidgetItem *child = GetDuplicateItem(ref); // create new child item for this group
                top->addChild(child); // add it to top item
                images[ref].group = group; // set current image's group to new group number
            }

            top->setExpanded(true); // expand this top item
        }
    }

    // clean duplicates list - should never happen
    /*for (int n = 0; n < ui->treeWidget_duplicates->topLevelItemCount(); n++) {
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(n);
        if (topItem->childCount() <= 1) {
            if (topItem->childCount() == 1) {
                int ref = topItem->child(0)->data(0, Qt::UserRole).toInt();
                images[ref].used = false;
                delete topItem->child(0);
            }
            delete topItem;
            n--;
        }
    }*/

    //// set color found or not in images list
    // green = used in a group, red = NOT used
    // we loose the information of file type color this way, but it helps looking for images to test further

    for (int n = 0; n < int(images.size()); n++) { // parse all images
        if ((!images[n].deleted) and (!images[n].error)) { // image valid ?
            if (images[n].used) // if it is used in a group
                images[n].imageItem->setBackground(QColor(0,255,0)); // color it in green in images list
            else // not used ?
                images[n].imageItem->setBackground(QColor(255,0,0)); // color it red
        }
    }

    //// sort list by resolution, descending in each group
    ui->treeWidget_duplicates->sortItems(4, Qt::DescendingOrder); // column 4 is the image resolution

    ShowDuplicatesListCount(); // show the number of matched images in duplicates list
}

int MainWindow::ShowDuplicatesListCount() // show number of images in duplicates list
{
    int countGroups = ui->treeWidget_duplicates->topLevelItemCount(); // count the nuber of groups in duplicates list
    ui->lcdNumber_nb_duplicates_groups->display(countGroups); // show this result

    QTreeWidgetItemIterator it(ui->treeWidget_duplicates); // iterator to parse all element of the QTreeWidget
    int countImages = 0; // counter for found images
    while (*it) { // all duplicates entry are tested
        countImages++; // count current one
        ++it; // next QTreeWidgetItem
    }
    countImages -= countGroups; // number of duplicates = number of items minus the group bars
    ui->lcdNumber_nb_duplicates_images->display(countImages); // display the result

    if (ui->treeWidget_duplicates->topLevelItemCount() == 0) { // if no group was found
        ui->label_no_duplicates->setVisible(true); // show the message
    }

    return countImages;
}

void MainWindow::SaveResults() // save results to csv file
{
    // write pairs to file
    std::locale mylocale("en_US.utf8");
    std::ofstream pairsFile("pairs.csv");
    pairsFile.imbue(mylocale);

    pairsFile << "Im1;Im2;Checksum;aHash;pHash;dHash;idHash;visHash;BlockMean;ColorMoments;MarrHildreth;Radial;DominantColors;Features;Homography;DNNClassify" << std::endl;

    std::map<cv::Point, struct_scores, struct_comparePoints>::iterator it = pairs.begin();
    while (it != pairs.end()) {
        pairsFile << it->first.x << ";" << it->first.y;
        for (int n = 0; n < img_similarity_count; n++)
             pairsFile << ";" << std::to_string(it->second.score[n]);
        pairsFile << std::endl;

        it++;
    }

    pairsFile.close();

    // write checksums to file
    std::ofstream hashesFile("hashes.csv");
    hashesFile.imbue(mylocale);

    hashesFile << "Path;Number;Hash" << std::endl;

    for (int n = 0; n < int(images.size()); n++) {
        if ((!images[n].deleted) and (!images[n].error)) {
            hashesFile << n << ";" << images[n].fullPath << ";";
            hashesFile << '"' << GetHashString(n, similarityAlgorithm) << '"';
            hashesFile << std::endl;
        }
    }

    hashesFile.close();
}

//// config files

bool MainWindow::ReadThresholdsConfig() // read thresholds from config file, return success as bool
{
    // config file read
    std::ifstream configFile; // open config file
    configfile::struct_config configData;
    bool go = configfile::OpenFile(configFile, "data/thresholds.cfg");
    int line = 0;

    if (!go) // problem was found while loading config file
        return false;

    // read lines of config file
    go = false; // indicate that parsing config file is still good, no errors
    while (configfile::ReadLine(configFile, configData)) { // read each line of config file
        line++; // current config file line number
        if (configData.type == "error") { // found an error before ?
            QMessageBox::critical(this, "Error",
                                        "Error in config file at line " + QString::number(line) + ":\n" + QString::fromStdString(configData.name));
            return false;
        }
        else if (configData.type == "value") { // found a value definition ?
            imageSimilarityAlgorithm similarityType = img_similarity_count;

            if (configData.valueType == "number") {
                switch (_(configData.name)) {
                    case _("checksum"):         similarityType = img_similarity_checksum; break;
                    case _("phash"):            similarityType = img_similarity_pHash; break;
                    case _("dhash"):            similarityType = img_similarity_dHash; break;
                    case _("idhash"):           similarityType = img_similarity_idHash; break;
                    case _("blockmean"):        similarityType = img_similarity_block_mean; break;
                    case _("marrhildreth"):     similarityType = img_similarity_marr_hildreth; break;
                    case _("radialvariance"):   similarityType = img_similarity_radial_variance; break;
                    //case _("colormoments"):     similarityType = img_similarity_color_moments; break;
                    case _("dominantcolors"):   similarityType = img_similarity_dominant_colors; break;
                    case _("features"):         similarityType = img_similarity_features; break;
                    case _("homography"):       similarityType = img_similarity_homography; break;
                    case _("dnnclassify"):      similarityType = img_similarity_dnn_classify; break;
                    case _("combined"):         similarityType = img_similarity_count; break;
                    default:                    QMessageBox::critical(this, "Error",
                                                                      "Error in config file at line " + QString::number(line) + ":\n" + "Unknown value type");
                                                break;
                }

                if (configData.valuesNumber.size() == 4) {
                    std::vector<float> values;
                    // categories : 0 < dissimilar < different < similar < ∞ (exact)
                    for (int i = 0; i < 4; i++)
                        thresholds[similarityType][i] = configData.valuesNumber[i];
                }
                else {
                    QMessageBox::critical(this, "Error",
                                                "Error in config file at line " + QString::number(line) + ":\n" + "4 values are expected");
                }
            }
        }
    } // end reading a line in config file

    return true;
}

void MainWindow::on_button_test_images_clicked()
{
    QPixmap q1, q2;
    cv::Mat im1, im2;

    for (int n = 0; n < ui->listWidget_image_list->count(); n++) {
        QListWidgetItem* item = ui->listWidget_image_list->item(n);
        if (item->checkState() == Qt::Checked) {
            int ref = item->data(Qt::UserRole).toInt();

            if (q1.isNull()) {
                q1 = LoadImagePix(images[ref].fullPath, images[ref].loadwith);
            }
            else if (q2.isNull()) {
                q2 = LoadImagePix(images[ref].fullPath, images[ref].loadwith);
                break;
            }
        }
    }

    if (q1.isNull())
        return;

    im1 = QPixmap2Mat(q1);
    if (!q2.isNull())
        im2 = QPixmap2Mat(q2);

    // code using im1 and im2 here
    /*//// NSFW test
    cv::dnn::Net dnn;
    dnn.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    dnn.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    dnn = cv::dnn::readNetFromCaffe("models/nsfw/nsfw.prototxt", "models/nsfw/nsfw.caffemodel");

    im1.convertTo(im1, CV_32FC3);
    cv::Mat blob = cv::dnn::blobFromImage(im1, 1.0, cv::Size(224, 224), cv::Scalar(104, 117, 123), false, false, CV_32F); // create blob imput - NSFW
    //cv::Mat blob = cv::dnn::blobFromImage(im1, 1.0, cv::Size(227, 227), cv::Scalar(112.005, 120.294, 138.682), false, false, CV_32F); // create blob imput - NSFW SqueezeNet

    // convolution -> return output
    cv::Mat output;
    dnn.setInput(blob);
    output =  dnn.forward("");

    output = output.reshape(1, 1);

    ui->doubleSpinBox_test->setValue(output.at<float>(1));*/

    std::vector<std::string> classes;
    classes.reserve(21850);
    std::ifstream file("models/imagenet-21k-classes.csv");
    std::string str;
    while (std::getline(file, str))
        classes.push_back(str);

    cv::dnn::Net dnn;
    dnn.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    dnn.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    dnn = cv::dnn::readNetFromCaffe("models/Inception21k-bn.prototxt", "models/Inception21k.caffemodel");

    im1.convertTo(im1, CV_32FC3);
    cv::Mat blob = cv::dnn::blobFromImage(im1, 1.0, cv::Size(224, 224), cv::Scalar(117, 117, 117), false, false, CV_32F); // create blob imput

    cv::Mat output;
    dnn.setInput(blob);
    output =  dnn.forward("");

    output = output.reshape(1, 1);

    int nbValues = 16;
    cv::Mat result = cv::Mat::zeros(2, nbValues, CV_32F);
    for (int n = 0; n < nbValues; n++) {
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;
        result.at<float>(0, n) = classId;
        result.at<float>(1, n) = confidence;
        output.at<float>(classId) = 0;
    }

    qDebug() << "-----------------------------------------------------";
    for (int n = 0; n < nbValues; n++)
        qDebug() << QString::fromStdString(classes[int(result.at<float>(0, n))]) << " " << result.at<float>(1, n) * 100.0f << "%";


    /*// show window with image
    if (!homography.empty()) {
        QPixmap pix = Mat2QPixmap(homography);
        int width = pix.width();
        int height = pix.height();
        if ((width > 1000) or (height > 1000))
            pix = pix.scaled(1000, 1000, Qt::KeepAspectRatio, Qt::SmoothTransformation);

        QSize screenSize = qApp->screens()[0]->size();

        QLabel *label_img = new QLabel (this);
        //label_img->setWindowModality(Qt::WindowModal);
        label_img->setWindowFlags(Qt::Window | Qt::WindowStaysOnTopHint);
        label_img->setWindowTitle(QString::number(width) + "x" + QString::number(height) + " - ");
        label_img->setGeometry((screenSize.width() - pix.width()) / 2, (screenSize.height() - pix.height()) / 2, pix.width(), pix.height());
        label_img->setPixmap(pix);
        label_img->show();
    }*/
}

void MainWindow::on_button_examine_clicked() // set as a test at the beginning, this function is useful to estimate the real visual similarity between two images
    // the first 2 checked items if the duplicates list are tested in this function
    // it shows in a new window the é images side by side, the one with least resolution is displayed on the left
    // "good" features matches are displayed with cyan lines
    // if an homography is found, draw it too
{
    //// find the 2 first checked images in duplicates list

    QPixmap q1, q2;
    int im1 = -1;
    int im2 = -1;

    for (int n = 0; n < ui->treeWidget_duplicates->topLevelItemCount(); n++) { // for each top-level item
        QTreeWidgetItem *topItem = ui->treeWidget_duplicates->topLevelItem(n); // current top-level item
        if (topItem->childCount() > 0) { // does it have children ? (should be yes all the time)
            for (int i = 0; i < int(topItem->childCount()); i++) { // parse all children of the top item
                 QTreeWidgetItem *child = topItem->child(i); // this child

                 int state = child->checkState(0); // is it checked ?
                 if (state == Qt::Checked) { // yes !
                     int ref = child->data(0, Qt::UserRole).toInt(); // get this child's index in internal images list

                     if (im1 == -1) { // is it the first image to test ?
                         im1 = ref;
                     }
                     else if (im2 == -1) { // is it the 2nd image to test ?
                         im2 = ref;
                         n = ui->treeWidget_duplicates->topLevelItemCount(); // end the main loop now that we have what we were looking for
                         break; // finished !
                     }
                 }
            }
        }
    }

    if ((im1 == -1) or (im2 == -1)) // at least two images checked !
        return;

    //// invert the images to have the smallest image on the left side
    if (images[im1].imageReducedGray.cols * images[im1].imageReducedGray.rows > images[im2].imageReducedGray.cols * images[im2].imageReducedGray.rows) {
        int tmp = im1;
        im1 = im2;
        im2 = tmp;
    }

    cv::Mat homography; // 3x3 resulting homography
    /*std::vector<cv::KeyPoint> keypoints1, keypoints2; // for image features
    cv::Mat descriptors1, descriptors2;*/
    std::vector<cv::Point2f> goodPoints1, goodPoints2;
    if (images[im1].keypoints.empty()) // for image 1 - if keypoints were not already computed
        ComputeImageDescriptors(images[im1].imageReducedGray, images[im1].keypoints, images[im1].descriptors, false, reducedSize, nbFeatures); // compute keypoints
    if (images[im2].keypoints.empty()) // same for image 2
        ComputeImageDescriptors(images[im2].imageReducedGray, images[im2].keypoints, images[im2].descriptors, false, reducedSize, nbFeatures);
    float score = 0; // score is est by default to the minimum
    homography = GetHomographyFromImagesFeatures(images[im1].imageReducedGray, images[im2].imageReducedGray,
                                                 images[im1].keypoints, images[im2].keypoints, images[im1].descriptors, images[im2].descriptors, goodPoints1, goodPoints2,
                                                 score, false, reducedSize, false, 0.8f, nbFeatures); // get the homography if it exists

    //// create double image view
    int width = images[im1].imageReduced.cols + images[im2].imageReduced.cols; // this image will contain the two images
    width += 200; // ... plus a border
    int height = std::max(images[im1].imageReduced.rows, images[im2].imageReduced.rows);
    height += 200;
    cv::Mat display = cv::Mat::zeros(height, width, CV_8UC3); // create the empty view, black background
    PasteImageColor(display, images[im1].imageReduced, 50, 100); // paste the reduced images on it
    PasteImageColor(display, images[im2].imageReduced, 150 + images[im1].imageReduced.cols, 100);

    //// draw the matches
    for (int n = 0; n < int(goodPoints1.size()); n++) { // from the "good matches" list
        goodPoints1[n] = goodPoints1[n] + cv::Point2f(50, 100); // add the needed shift to the coordinates
        goodPoints2[n] = goodPoints2[n] + cv::Point2f(150, 100);
        cv::line(display, goodPoints1[n], goodPoints2[n] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(255, 255, 0), 1, cv::LINE_AA); // draw the line in cyan
    }

    // show homography
    if (score > 0) { // a positive score means that a homography was found
        // get the corners from the image 1 ( the object to be "detected" )
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cv::Point(0,0);
        obj_corners[1] = cv::Point(images[im1].imageReduced.cols, 0);
        obj_corners[2] = cv::Point(images[im1].imageReduced.cols, images[im1].imageReduced.rows);
        obj_corners[3] = cv::Point(0, images[im1].imageReduced.rows);
        std::vector<cv::Point2f> scene_corners(4);
        // transform these corners with the homography matrix
        cv::perspectiveTransform(obj_corners, scene_corners, homography);
        // add the needed shift to draw it on the right pimage
        for (int n = 0; n < 4; n++) {
            obj_corners[n] = obj_corners[n] + cv::Point2f(50, 100);
            scene_corners[n] = scene_corners[n] + cv::Point2f(150, 100);
        }
        // draw the lines between the original corners - green
        cv::line(display, obj_corners[0], obj_corners[1], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::line(display, obj_corners[1], obj_corners[2], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::line(display, obj_corners[2], obj_corners[3], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::line(display, obj_corners[3], obj_corners[0], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        // draw the lines between the original corners and the projected corners - yellow
        cv::arrowedLine(display, obj_corners[0], scene_corners[0] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0, 0.01);
        cv::arrowedLine(display, obj_corners[1], scene_corners[1] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0, 0.01);
        cv::arrowedLine(display, obj_corners[2], scene_corners[2] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0, 0.01);
        cv::arrowedLine(display, obj_corners[3], scene_corners[3] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0, 0.01);
        // draw the lines between the corners (the projected object) - red
        cv::line(display, scene_corners[0] + cv::Point2f( images[im1].imageReduced.cols, 0), scene_corners[1] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::line(display, scene_corners[1] + cv::Point2f( images[im1].imageReduced.cols, 0), scene_corners[2] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::line(display, scene_corners[2] + cv::Point2f( images[im1].imageReduced.cols, 0), scene_corners[3] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::line(display, scene_corners[3] + cv::Point2f( images[im1].imageReduced.cols, 0), scene_corners[0] + cv::Point2f(images[im1].imageReduced.cols, 0), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    //// show result on screen

    // size of the widget
    QPixmap pix = Mat2QPixmap(display);
    width = pix.width();
    height = pix.height();
    if ((width > 1000) or (height > 1000))
        pix = pix.scaled(1000, 1000, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // info for window title
    auto pairScore = pairs.find(OrderedPair(im1, im2));
    score = pairScore->second.score[similarityAlgorithm];

    QSize screenSize = qApp->screens()[0]->size();

    QLabel *label_img = new QLabel (this);

    // display a new window containing the result
    label_img->setWindowFlags(Qt::Window | Qt::WindowStaysOnTopHint);
    label_img->setWindowTitle("1: " + QString::number(images[im1].width) + "x" + QString::number(images[im1].height)
                              + " - 2: " + QString::number(images[im2].width) + "x" + QString::number(images[im2].height)
                              + " - Similarity= " + QString::number(score) + "%");
    label_img->setGeometry((screenSize.width() - pix.width()) / 2, (screenSize.height() - pix.height()) / 2, pix.width(), pix.height());
    label_img->setPixmap(pix);
    label_img->show();
}




