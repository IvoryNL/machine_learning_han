#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_sourcePathEdit_textChanged(const QString &arg1);

    void on_destinationPathEdit_textChanged(const QString &arg1);

    void on_lowerHueSlider_valueChanged(int value);

    void on_lowerSaturationSlider_valueChanged(int value);

    void on_lowerValueSlider_valueChanged(int value);

    void on_upperHueSlider_valueChanged(int value);

    void on_upperSaturationSlider_valueChanged(int value);

    void on_upperValueSlider_valueChanged(int value);

    void on_processImagesButton_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
