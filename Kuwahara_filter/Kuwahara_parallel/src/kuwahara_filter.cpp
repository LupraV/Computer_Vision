#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;
using namespace std::chrono;

void computeIntegralImages(const Mat& src, Mat& integralImg, Mat& integralImgSq) {
    integral(src, integralImg, integralImgSq, CV_64F);
}

class ParallelKuwahara : public ParallelLoopBody {
private:
    const Mat& src;
    Mat& dst;
    const Mat& integralImg;
    const Mat& integralImgSq;
    int kernelSize;
    int halfKernel;

public:
    ParallelKuwahara(const Mat& src, Mat& dst, const Mat& integralImg, const Mat& integralImgSq, int kernelSize)
        : src(src), dst(dst), integralImg(integralImg), integralImgSq(integralImgSq), kernelSize(kernelSize), halfKernel(kernelSize / 2) {}

    void operator()(const Range& range) const override {
        for (int r = range.start; r < range.end; r++) {
            int y = r / src.cols;
            int x = r % src.cols;

            double minVar = numeric_limits<double>::max();
            double bestMean = 0;

            for (int dy = -halfKernel; dy <= 0; dy += halfKernel) {
                for (int dx = -halfKernel; dx <= 0; dx += halfKernel) {
                    int x1 = max(x + dx + 1, 1);
                    int y1 = max(y + dy + 1, 1);
                    int x2 = min(x + dx + halfKernel + 1, src.cols);
                    int y2 = min(y + dy + halfKernel + 1, src.rows);

                    double sum = integralImg.at<double>(y2, x2) - integralImg.at<double>(y2, x1 - 1) - integralImg.at<double>(y1 - 1, x2) + integralImg.at<double>(y1 - 1, x1 - 1);
                    double sqSum = integralImgSq.at<double>(y2, x2) - integralImgSq.at<double>(y2, x1 - 1) - integralImgSq.at<double>(y1 - 1, x2) + integralImgSq.at<double>(y1 - 1, x1 - 1);

                    double area = (x2 - x1 + 1) * (y2 - y1 + 1);
                    double mean = sum / area;
                    double variance = (sqSum / area) - (mean * mean);

                    if (variance < minVar) {
                        minVar = variance;
                        bestMean = mean;
                    }
                }
            }

            dst.at<uchar>(y, x) = saturate_cast<uchar>(bestMean);
        }
    }
};

Mat applyKuwaharaFilter(const Mat& src, int kernelSize) {
    if (kernelSize % 2 == 0 || kernelSize < 3 || kernelSize > 15) {
        cerr << "Note: Kernel size must be an odd number between 3 and 15." << endl;
        return Mat();
    }

    Mat srcGray = src.clone();
    if (srcGray.channels() > 1) {
        cvtColor(srcGray, srcGray, COLOR_BGR2GRAY);
    }

    Mat integralImg, integralImgSq;
    computeIntegralImages(srcGray, integralImg, integralImgSq);

    Mat dst = Mat::zeros(srcGray.size(), srcGray.type());
    ParallelKuwahara parallelKuwahara(srcGray, dst, integralImg, integralImgSq, kernelSize);
    parallel_for_(Range(0, srcGray.rows * srcGray.cols), parallelKuwahara);

    return dst;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <InputImage> <OutputImage> <KernelSize>" << endl;
        return -1;
    }

    String inputPath = argv[1];
    String outputPath = argv[2];
    int kernelSize = stoi(argv[3]);

    Mat src = imread(inputPath, IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Error loading image!" << endl;
        return -1;
    }

    auto start = high_resolution_clock::now();
    Mat dst = applyKuwaharaFilter(src, kernelSize);
    auto stop = high_resolution_clock::now();

    if (dst.empty()) {
        cerr << "Failed to apply Kuwahara filter!" << endl;
        return -1;
    }

    if (!imwrite(outputPath, dst)) {
        cerr << "Failed to save output image!" << endl;
        return -1;
    }

    auto duration_ns = duration_cast<nanoseconds>(stop - start);
    auto duration_ms = duration_cast<milliseconds>(duration_ns);
    //cout << "Kuwahara filter flawlessly executed (almost)! Execution time: " << duration.count() << " ns" << endl;
    cout << "Lightning fast flawless Kuwahara filter (almost)! Execution time: " << duration_ns.count() << " ns (" << duration_ms.count() << " ms)" << endl;

    return 0;
}