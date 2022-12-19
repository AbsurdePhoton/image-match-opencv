#-------------------------------------------------
#
#   ImageMatch: Find Image Duplicates with openCV
#
#    by AbsurdePhoton - www.absurdephoton.fr
#
#                v0 - 2022/10/22
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = image-match
TEMPLATE = app

#INCLUDEPATH += /usr/local/include/opencv4/opencv2
#INCLUDEPATH += /usr/include/eigen3

LIBS += -fopenmp -ltbb -ffast-math -fno-math-errno

SOURCES +=  main.cpp\
            mainwindow.cpp \
            lib/image-utils.cpp \
            lib/dominant-colors.cpp \
            lib/color-spaces.cpp \
            lib/angles.cpp \
            lib/config-file.cpp \
            #lib/csv-file.cpp \
            lib/string-utils.cpp \
            lib/contours.cpp \
            #lib/image-effects.cpp \
            lib/image-transform.cpp \
            lib/image-color.cpp \
            lib/image-compare.cpp \
            #lib/image-filter.cpp \
            #lib/image-draw.cpp \
            #lib/image-lut.cpp \
            #lib/EDLine/EDColor.cpp \
            #lib/EDLine/ED.cpp \
            #lib/polypartition/polypartition.cpp \
            #widgets/image-viewer.cpp \
            #widgets/dial-range.cpp \
            widgets/file-dialog.cpp


HEADERS  += mainwindow.h \
            lib/image-utils.h \
            lib/dominant-colors.h \
            lib/color-spaces.h \
            lib/angles.h \
            lib/config-file.h \
            #lib/csv-file.h \
            lib/string-utils.h \
            lib/contours.h \
            #lib/image-effects.h \
            lib/image-transform.h \
            lib/image-color.h \
            lib/image-compare.h \
            #lib/image-filter.h \
            #lib/image-draw.h \
            #lib/image-lut.h \
            lib/randomizer.h \
            #lib/EDLine/EDColor.h \
            #lib/EDLine/ED.h \
            #lib/polypartition/polypartition.h \
            #widgets/image-viewer.h \
            #widgets/dial-range.h \
            widgets/file-dialog.h


FORMS    += mainwindow.ui

# add the package opencv to pkg-config
CONFIG += link_pkgconfig
PKGCONFIG += opencv4

# icons
RESOURCES += resources.qrc

CONFIG += c++17

# openOMP
QMAKE_LFLAGS += -fopenmp
QMAKE_CXXFLAGS += -fopenmp
QMAKE_CXXFLAGS += -ffast-math -msse4.2 -march=native -ftree-vectorize

# optimization level
#QMAKE_CXXFLAGS_RELEASE -= -O2
#QMAKE_CXXFLAGS_RELEASE *= -O3
