#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace std::filesystem;

QString sourceFilePath;
QString destinationFilePath;
QString status;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_sourcePathEdit_textChanged(const QString &arg1)
{
    sourceFilePath = arg1;

    ui->statusLabel->setText("...");
}


void MainWindow::on_destinationPathEdit_textChanged(const QString &arg1)
{
    destinationFilePath = arg1;

    ui->statusLabel->setText("...");
}

void MainWindow::on_processImagesButton_clicked()
{
    string sourcePath = sourceFilePath.toStdString();
    string outputPath = destinationFilePath.toStdString();
    int counter = 0;

    for (const auto & entry : directory_iterator(sourcePath))
    {
        if(entry.is_directory())
        {
            continue;
        }

        Mat src = imread(entry.path().string(), IMREAD_COLOR);

        if (!src.data)
        {
            cout << "Error loading image";
            return;
        }

        Mat lab;
        cvtColor(src, lab, COLOR_BGR2Lab);

        vector<Mat> labChannels(3);
        split(lab, labChannels);
        Mat L = labChannels[1];

        Mat thresh;
        threshold(L, thresh, 0, 235, THRESH_BINARY + THRESH_OTSU);

        Mat resultOpen;
        Mat kernelOpen = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
        morphologyEx(thresh, resultOpen, MORPH_OPEN, kernelOpen);

        Mat resultClose;
        Mat kernelClose = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
        morphologyEx(thresh, resultClose, MORPH_CLOSE, kernelClose);

        Mat morphedResult = Mat::zeros(thresh.rows, thresh.cols, thresh.type());
        bitwise_or(resultOpen, resultClose, morphedResult);

        Mat finalResult;
        kernelOpen = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
        morphologyEx(morphedResult, finalResult, MORPH_OPEN, kernelOpen);

        if (!exists(outputPath)) {
            if (!create_directory(outputPath)) {
                cerr << "Error: Could not create the directory!" << endl;
                return;
            }
        }
        string filename = entry.path().filename().string();
        string output_file_path = outputPath + "/" + filename;

        if (!imwrite(output_file_path, finalResult)) {
            cerr << "Error: Could not save the image!" << endl;
            return;
        }

        ui->statusLabel->setText(QString::fromStdString(to_string(++counter)));
    }
    ui->statusLabel->setText("Finished processing");
}

