// Luis Vieira - 23012096

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <limits>

using namespace cv;
using namespace std;
using namespace std::chrono;

// Calculating integral & sq. integral imgage
void calcIntegralImg(const Mat& src, Mat& integralImg, Mat& integralSqImg) {
    integral(src, integralImg, integralSqImg, CV_64F);
}

// Precomputing Kuwahara using pre-calculated integral image to pass it to Kuwahara filter function
void precompIntegralImages(const Mat& src, Mat& dst, const Mat& integralImg, const Mat& integralSqImg, int kernelSize) {
    int halfKernel = kernelSize / 2;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            double minVar = numeric_limits<double>::max();
            double bestMean = 0;

            // checking all quadrants around pixel for min var
            for (int dy = -halfKernel; dy <= 0; dy += halfKernel) {
                int y1 = max(y + dy + 1, 1);
                int y2 = min(y + dy + halfKernel + 1, src.rows);

                for (int dx = -halfKernel; dx <= 0; dx += halfKernel) {
                    int x1 = max(x + dx + 1, 1);
                    int x2 = min(x + dx + halfKernel + 1, src.cols);

                    // Calculating sum, sqSum, area, mean, var for quadrant
                    double sum, sqSum, area, mean, variance;
                    sum = integralImg.at<double>(y2, x2) - integralImg.at<double>(y2, x1 - 1) - integralImg.at<double>(y1 - 1, x2) + integralImg.at<double>(y1 - 1, x1 - 1);
                    sqSum = integralSqImg.at<double>(y2, x2) - integralSqImg.at<double>(y2, x1 - 1) - integralSqImg.at<double>(y1 - 1, x2) + integralSqImg.at<double>(y1 - 1, x1 - 1);
                    area = (x2 - x1 + 1) * (y2 - y1 + 1);
                    mean = sum / area;
                    variance = (sqSum / area) - (mean * mean);
                    
                    if (variance < minVar) {
                        minVar = variance;
                        bestMean = mean;
                    }
                }
            }
            dst.at<uchar>(y, x) = saturate_cast<uchar>(bestMean);
        }
    }
}

// Applying Kuawahara filter based on kernel size
Mat KuwaharaFilter(const Mat& src, int kernelSize) {
    if (kernelSize % 2 == 0 || kernelSize < 3 || kernelSize > 15) {
        cerr << "Note: Kernel size must be odd number between 3 & 15." << endl;
        return Mat();
    }

    Mat integralImg, integralSqImg;
    calcIntegralImg(src, integralImg, integralSqImg);

    Mat dst = Mat::zeros(src.size(), src.type());
    precompIntegralImages(src, dst, integralImg, integralSqImg, kernelSize); // applying filter

    return dst;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <InputImage> <OutputImage> <KernelSize>" << endl;
        return -1;
    }

    // loading image to apply filter
    String inputPath = argv[1];
    String outputPath = argv[2];
    int kernelSize = stoi(argv[3]); //convert kernel size to int

    Mat src = imread(inputPath, IMREAD_GRAYSCALE); // loading as greyscale
    if (src.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    auto start = high_resolution_clock::now();
    Mat dst = KuwaharaFilter(src, kernelSize);
    auto end = high_resolution_clock::now();

    if (dst.empty()) {
        cerr << "Kuwahara filter failed!" << endl;
        return -1;
    }
    if (!imwrite(outputPath, dst)) {
        cerr << "Failed saving image!" << endl;
        return -1;
    }

    auto duration_ns = duration_cast<nanoseconds>(end - start);
    auto duration_ms = duration_cast<milliseconds>(duration_ns);
    cout << "Execution time: " << duration_ns.count() << " ns (" << duration_ms.count() << " ms)" << endl;

    return 0;
}