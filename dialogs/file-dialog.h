/*#-------------------------------------------------
#
#    Qt file dialog widget with image preview
#
#    by AbsurdePhoton - www.absurdephoton.fr
#
#               v1.1 - 2025/05/08
#
#-------------------------------------------------*/

#ifndef PREVIEW_FILE_DIALOG_H
#define PREVIEW_FILE_DIALOG_H

#include <QFileDialog>


static const QString previewImageCommon   = "*.jpg *.jpeg *.jp2 *.jpe *.jxl *.png *.tif *.tiff *.webp *.avif *.gif";
static const QString previewImageLossless = "*.png *.tif *.tiff";
static const QString previewImageLossy    = "*.jpg *.jpeg *.jp2 *.jpe *.jxl *.webp *.avif";

class QLabel;

class PreviewFileDialog : public QFileDialog
{
    Q_OBJECT

public:
    explicit PreviewFileDialog( QWidget* parent = 0,
                                const QString & caption = QString(), // window caption
                                const QString & directory = QString(), // directory to start in
                                const QString & filter = QString(), // files filter(s)
                                const bool &save = false
                              ); // save=true for saving a file
    virtual ~PreviewFileDialog() {}

    QString GetSelectedFile(); // return selected file path

protected slots:
    void OnCurrentChanged(const QString & path); // show thumbnail when a file is selected
    void OnRejected(); // if Cancel is clicked filename is empty

protected:
    QLabel* mpPreview; // label to show the thumbnail
    QString selectedFilename; // keep selected filename path
    bool canceled;

};

#endif // PREVIEW_FILE_DIALOG_H
