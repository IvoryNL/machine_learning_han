/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.7.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QLabel *label;
    QLineEdit *sourcePathEdit;
    QLineEdit *destinationPathEdit;
    QLabel *label_2;
    QPushButton *processImagesButton;
    QLabel *label_11;
    QLabel *statusLabel;
    QLabel *label_13;
    QSlider *upperHueSlider;
    QSlider *upperValueSlider;
    QLabel *lowerValueLabel;
    QLabel *upperValueLabel;
    QSlider *lowerSaturationSlider;
    QLabel *upperSaturationLabel;
    QLabel *label_14;
    QLabel *label_15;
    QLabel *label_16;
    QSlider *lowerHueSlider;
    QLabel *lowerHueLabel;
    QLabel *upperHueLabel;
    QLabel *label_17;
    QLabel *label_18;
    QLabel *label_19;
    QSlider *lowerValueSlider;
    QSlider *upperSaturationSlider;
    QLabel *lowerSaturationLabel;
    QLabel *label_20;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(1250, 1159);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        label = new QLabel(centralwidget);
        label->setObjectName("label");
        label->setGeometry(QRect(60, 30, 411, 41));
        QFont font;
        font.setPointSize(16);
        font.setBold(true);
        label->setFont(font);
        sourcePathEdit = new QLineEdit(centralwidget);
        sourcePathEdit->setObjectName("sourcePathEdit");
        sourcePathEdit->setGeometry(QRect(60, 90, 1081, 51));
        QFont font1;
        font1.setPointSize(14);
        sourcePathEdit->setFont(font1);
        destinationPathEdit = new QLineEdit(centralwidget);
        destinationPathEdit->setObjectName("destinationPathEdit");
        destinationPathEdit->setGeometry(QRect(60, 220, 1081, 51));
        destinationPathEdit->setFont(font1);
        label_2 = new QLabel(centralwidget);
        label_2->setObjectName("label_2");
        label_2->setGeometry(QRect(60, 160, 411, 41));
        label_2->setFont(font);
        processImagesButton = new QPushButton(centralwidget);
        processImagesButton->setObjectName("processImagesButton");
        processImagesButton->setGeometry(QRect(60, 900, 1081, 71));
        processImagesButton->setFont(font);
        label_11 = new QLabel(centralwidget);
        label_11->setObjectName("label_11");
        label_11->setGeometry(QRect(60, 1010, 101, 41));
        label_11->setFont(font);
        statusLabel = new QLabel(centralwidget);
        statusLabel->setObjectName("statusLabel");
        statusLabel->setGeometry(QRect(190, 1010, 411, 41));
        statusLabel->setFont(font);
        label_13 = new QLabel(centralwidget);
        label_13->setObjectName("label_13");
        label_13->setGeometry(QRect(120, 810, 81, 20));
        label_13->setFont(font);
        upperHueSlider = new QSlider(centralwidget);
        upperHueSlider->setObjectName("upperHueSlider");
        upperHueSlider->setGeometry(QRect(220, 680, 551, 20));
        upperHueSlider->setMaximum(255);
        upperHueSlider->setOrientation(Qt::Orientation::Horizontal);
        upperValueSlider = new QSlider(centralwidget);
        upperValueSlider->setObjectName("upperValueSlider");
        upperValueSlider->setGeometry(QRect(220, 820, 551, 18));
        upperValueSlider->setMaximum(255);
        upperValueSlider->setOrientation(Qt::Orientation::Horizontal);
        lowerValueLabel = new QLabel(centralwidget);
        lowerValueLabel->setObjectName("lowerValueLabel");
        lowerValueLabel->setGeometry(QRect(790, 540, 63, 20));
        lowerValueLabel->setFont(font);
        upperValueLabel = new QLabel(centralwidget);
        upperValueLabel->setObjectName("upperValueLabel");
        upperValueLabel->setGeometry(QRect(790, 820, 63, 20));
        upperValueLabel->setFont(font);
        lowerSaturationSlider = new QSlider(centralwidget);
        lowerSaturationSlider->setObjectName("lowerSaturationSlider");
        lowerSaturationSlider->setGeometry(QRect(220, 470, 551, 18));
        lowerSaturationSlider->setMaximum(255);
        lowerSaturationSlider->setOrientation(Qt::Orientation::Horizontal);
        upperSaturationLabel = new QLabel(centralwidget);
        upperSaturationLabel->setObjectName("upperSaturationLabel");
        upperSaturationLabel->setGeometry(QRect(790, 750, 63, 20));
        upperSaturationLabel->setFont(font);
        label_14 = new QLabel(centralwidget);
        label_14->setObjectName("label_14");
        label_14->setGeometry(QRect(400, 320, 181, 41));
        label_14->setFont(font);
        label_15 = new QLabel(centralwidget);
        label_15->setObjectName("label_15");
        label_15->setGeometry(QRect(60, 470, 141, 20));
        label_15->setFont(font);
        label_16 = new QLabel(centralwidget);
        label_16->setObjectName("label_16");
        label_16->setGeometry(QRect(60, 750, 141, 20));
        label_16->setFont(font);
        lowerHueSlider = new QSlider(centralwidget);
        lowerHueSlider->setObjectName("lowerHueSlider");
        lowerHueSlider->setGeometry(QRect(220, 400, 551, 20));
        lowerHueSlider->setMaximum(255);
        lowerHueSlider->setOrientation(Qt::Orientation::Horizontal);
        lowerHueLabel = new QLabel(centralwidget);
        lowerHueLabel->setObjectName("lowerHueLabel");
        lowerHueLabel->setGeometry(QRect(790, 400, 63, 20));
        lowerHueLabel->setFont(font);
        upperHueLabel = new QLabel(centralwidget);
        upperHueLabel->setObjectName("upperHueLabel");
        upperHueLabel->setGeometry(QRect(790, 680, 63, 20));
        upperHueLabel->setFont(font);
        label_17 = new QLabel(centralwidget);
        label_17->setObjectName("label_17");
        label_17->setGeometry(QRect(130, 400, 63, 20));
        label_17->setFont(font);
        label_18 = new QLabel(centralwidget);
        label_18->setObjectName("label_18");
        label_18->setGeometry(QRect(120, 530, 81, 20));
        label_18->setFont(font);
        label_19 = new QLabel(centralwidget);
        label_19->setObjectName("label_19");
        label_19->setGeometry(QRect(410, 600, 181, 41));
        label_19->setFont(font);
        lowerValueSlider = new QSlider(centralwidget);
        lowerValueSlider->setObjectName("lowerValueSlider");
        lowerValueSlider->setGeometry(QRect(220, 540, 551, 18));
        lowerValueSlider->setMaximum(255);
        lowerValueSlider->setOrientation(Qt::Orientation::Horizontal);
        upperSaturationSlider = new QSlider(centralwidget);
        upperSaturationSlider->setObjectName("upperSaturationSlider");
        upperSaturationSlider->setGeometry(QRect(220, 750, 551, 18));
        upperSaturationSlider->setMaximum(255);
        upperSaturationSlider->setOrientation(Qt::Orientation::Horizontal);
        lowerSaturationLabel = new QLabel(centralwidget);
        lowerSaturationLabel->setObjectName("lowerSaturationLabel");
        lowerSaturationLabel->setGeometry(QRect(790, 470, 63, 20));
        lowerSaturationLabel->setFont(font);
        label_20 = new QLabel(centralwidget);
        label_20->setObjectName("label_20");
        label_20->setGeometry(QRect(130, 680, 63, 20));
        label_20->setFont(font);
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 1250, 25));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "Source path", nullptr));
        sourcePathEdit->setText(QString());
        destinationPathEdit->setText(QString());
        label_2->setText(QCoreApplication::translate("MainWindow", "Destination path", nullptr));
        processImagesButton->setText(QCoreApplication::translate("MainWindow", "Process images", nullptr));
        label_11->setText(QCoreApplication::translate("MainWindow", "Status:", nullptr));
        statusLabel->setText(QCoreApplication::translate("MainWindow", "...", nullptr));
        label_13->setText(QCoreApplication::translate("MainWindow", "Value", nullptr));
        lowerValueLabel->setText(QCoreApplication::translate("MainWindow", "...", nullptr));
        upperValueLabel->setText(QCoreApplication::translate("MainWindow", "...", nullptr));
        upperSaturationLabel->setText(QCoreApplication::translate("MainWindow", "...", nullptr));
        label_14->setText(QCoreApplication::translate("MainWindow", "Lower HSV", nullptr));
        label_15->setText(QCoreApplication::translate("MainWindow", "Saturation", nullptr));
        label_16->setText(QCoreApplication::translate("MainWindow", "Saturation", nullptr));
        lowerHueLabel->setText(QCoreApplication::translate("MainWindow", "...", nullptr));
        upperHueLabel->setText(QCoreApplication::translate("MainWindow", "...", nullptr));
        label_17->setText(QCoreApplication::translate("MainWindow", "Hue", nullptr));
        label_18->setText(QCoreApplication::translate("MainWindow", "Value", nullptr));
        label_19->setText(QCoreApplication::translate("MainWindow", "Upper HSV", nullptr));
        lowerSaturationLabel->setText(QCoreApplication::translate("MainWindow", "...", nullptr));
        label_20->setText(QCoreApplication::translate("MainWindow", "Hue", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
