// Luis Vieira - 23012096

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <utility>
#include <map>
#include <limits>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

// encoding from decoding.h
char encodingarray[64]={' ','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','x','y','w','z',
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','X','Y','W','Z',
    '0','1','2','3','4','5','6','7','8','9','.'};

// Functions
std::pair<vector<Vec3f>, vector<Vec3f>> detectCircles(const Mat& src);
vector<Point2f> avgCenters(const vector<Vec3f>& circles1, const vector<Vec3f>& circles2);
vector<Vec3f> getCircles(const vector<Vec3f>& circles, float thresh);
vector<Point2f> getCircleCenters(const vector<Vec3f>& circles);
double euclideanDist(const Point2f& a, const Point2f& b);
std::tuple<Mat, Mat, vector<Point2f>, Point2f> rotateQR(const Mat& processedImg, const Mat& srcImg, const vector<Point2f>& averagedCenters);
Mat cropScaleQR(const Mat& rotatedImgColor, const vector<Point2f>& rotatedCenters);
int colorToBinary(const Vec3b& color);
char binaryToChar(int binaryValue);
bool compareVec3b(const Vec3b &a, const Vec3b &b);
double OtsuThresh(const Mat& src);
Vec3b blockColor(const Mat& img, const Point& center, int blockSize, double OtsuTresh);
Vec3b clampColor(const Vec3b& color, double OtsuTresh);
std::string decodeQR(const Mat& croppedScaledQR, double OtsuTresh);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Detecting circles using Hough transform, both HSV (blue) and greyscale (+ Gaussian blur).
// Using Otsu threshold to account for lighting extremes
// Uses greyscale for detection, if not tries detecting blue using hsv.
// averages their closest centers x,y (eucleadian distance). If only finds hsv or grey, uses those.

std::pair<vector<Vec3f>, vector<Vec3f>> detectCircles(const Mat& src) {
    vector<Vec3f> hsvCircles, greyCircles;
    // Otsu adjusting histogram lighting
    double OtsuThreshold = OtsuThresh(src);

    // HSV Circle Detection
    Mat hsvImg, mask;
    cvtColor(src, hsvImg, COLOR_BGR2HSV);
    Scalar lower_blue(100, 10, 10), upper_blue(140, 255, 255);
    inRange(hsvImg, lower_blue, upper_blue, mask);
    HoughCircles(mask, hsvCircles, HOUGH_GRADIENT_ALT, 1, mask.rows/8, 150, .7, 2, mask.rows/6); 

    // greyscale circle Detection
    Mat processedImg;
    cvtColor(src, processedImg, COLOR_BGR2GRAY);
    threshold(processedImg, processedImg, OtsuThreshold, 255, THRESH_BINARY | THRESH_OTSU);
    GaussianBlur(processedImg, processedImg, Size(5, 5), 2, 2);
    //medianBlur(processedImg, processedImg, 7);
    HoughCircles(processedImg, greyCircles, HOUGH_GRADIENT_ALT, 1, processedImg.rows/8, 200, .9, 3, processedImg.rows/6);

    // neighbouring circles, ignore false positives
    float thresh = src.rows / 20.0f;
    hsvCircles = getCircles(hsvCircles, thresh);
    greyCircles = getCircles(greyCircles, thresh);

    return std::make_pair(hsvCircles, greyCircles);
}

vector<Vec3f> getCircles(const vector<Vec3f>& circles, float thresh) {
    map<pair<int, int>, Vec3f> uniqueCircles;
    for (const Vec3f& circle : circles) {
        Point2f center(circle[0], circle[1]);
        bool isConcentric = false;
        for (auto& uc : uniqueCircles) {
            if (euclideanDist(center, Point2f(uc.second[0], uc.second[1])) < thresh) {
                isConcentric = true;
                if (circle[2] > uc.second[2]) {
                    uc.second = circle;
                }
                break;
            }
        }
        if (!isConcentric) {
            uniqueCircles[{static_cast<int>(center.x), static_cast<int>(center.y)}] = circle;
        }
    }
    vector<Vec3f> concentricCircles;
    for (const auto& uc : uniqueCircles) {
        concentricCircles.push_back(uc.second);
    }
    return concentricCircles;
}

vector<Point2f> avgCenters(const vector<Vec3f>& circles1, const vector<Vec3f>& circles2) {
    vector<Point2f> centers1 = getCircleCenters(circles1);
    vector<Point2f> centers2 = getCircleCenters(circles2);
    vector<Point2f> averagedCenters;

    if (centers1.empty() || centers2.empty()) return {};

    for (auto& center1 : centers1) {
        float minDistance = std::numeric_limits<float>::max();
        Point2f closestCenter;

        // closest center in grey to center in hsv
        for (auto& center2 : centers2) {
            float distance = euclideanDist(center1, center2);
            if (distance < minDistance) {
                minDistance = distance;
                closestCenter = center2;
            }
        }
        // avg closest centers
        Point2f avgCenter((center1.x + closestCenter.x) / 2.0f, (center1.y + closestCenter.y) / 2.0f);
        averagedCenters.push_back(avgCenter);
    }
    return averagedCenters;
}

vector<Point2f> getCircleCenters(const vector<Vec3f>& circles) {
    vector<Point2f> centers;
    for (const auto& circle : circles) {
        centers.emplace_back(circle[0], circle[1]);
    }
    return centers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Rotating qr image function (Affine transform):
// Uses the hypotenuse mid point (qr center) of the triangle formed by the centers for rotation.
// Rotate based on angle, using the outlier center (center not on the hypotenuse - bottom left corner of qr),
// get angle formed with mid point (qr/hypotenuse center )
// Finally crop&scale it to 940x940px, so that each of qr's grid 47x47 blocks becomes 20x20px
// Scaling is based on hypotenuse length and adjusted by 47/41 (total height/width blocks / center's height/width block dist)
// 
// To do: implement Canny, contours and wrapTransform to detect qr boundary box and adjust for perspective!

// calculate hypotenuse length (circles' triangle)
double euclideanDist(const Point2f& a, const Point2f& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Rotate qr image based on detected circles' triangle hypotenuse (not great but works!)
std::tuple<Mat, Mat, vector<Point2f>, Point2f> rotateQR(const Mat& processedImg, const Mat& srcImg, const vector<Point2f>& averagedCenters) {
    if (averagedCenters.size() != 3) throw std::runtime_error("Need 3 circles for rotation!");

    double maxDist = 0;
    std::pair<Point2f, Point2f> hypotenusePoints;
    Point2f outlierCircle;  // circle not on hypotenuse

    // find hypotenuse and outlier circle
    for (size_t i = 0; i < averagedCenters.size(); ++i) {
        for (size_t j = i + 1; j < averagedCenters.size(); ++j) {
            double dist = euclideanDist(averagedCenters[i], averagedCenters[j]);
            if (dist > maxDist) {
                maxDist = dist;
                hypotenusePoints.first = averagedCenters[i];
                hypotenusePoints.second = averagedCenters[j];
            }
        }
    }
    for (const auto& point : averagedCenters) {
        if (point != hypotenusePoints.first && point != hypotenusePoints.second) {
            outlierCircle = point;
            break;
        }
    }
    // Get center of hypotenuse/qr
    Point2f midPoint = (hypotenusePoints.first + hypotenusePoints.second) / 2;
    double angle = atan2(outlierCircle.y - midPoint.y, outlierCircle.x - midPoint.x) * 180 / CV_PI;
    Mat rotatedImgColor, rotatedImg, rotMatrix;

    if (outlierCircle.y < midPoint.y || angle != -135) { //angle from hypotenuse center to outlier circle
        rotMatrix = getRotationMatrix2D(midPoint, -135 + angle, 1.0);
        warpAffine(processedImg, rotatedImg, rotMatrix, processedImg.size());
        warpAffine(srcImg, rotatedImgColor, rotMatrix, srcImg.size());
    } else {
        rotatedImgColor = srcImg.clone();
        rotatedImg = processedImg.clone();
    }

    vector<Point2f> rotatedPoints(3);
    transform(averagedCenters.begin(), averagedCenters.end(), rotatedPoints.begin(),
              [&](const Point2f &pt) -> Point2f {
                  Mat ptMat = (Mat_<double>(3, 1) << pt.x, pt.y, 1);
                  Mat rotatedPtMat = rotMatrix * ptMat;
                  return Point2f(rotatedPtMat.at<double>(0, 0), rotatedPtMat.at<double>(1, 0));
              });

    return std::make_tuple(rotatedImgColor, rotatedImg, rotatedPoints, midPoint);
}

// Crop & scale to 940x940 using circles centers
Mat cropScaleQR(const Mat& rotatedImgColor, const vector<Point2f>& rotatedCenters) {
    if (rotatedCenters.size() != 3) {
        cerr << "Need 3 circles for cropping!" << endl;
        return Mat();
    }

    // hypotenuse length
    double hypotenuseLength = 0;
    Point2f p1, p2;
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            double currentDistance = norm(rotatedCenters[i] - rotatedCenters[j]);
            if (currentDistance > hypotenuseLength) {
                hypotenuseLength = currentDistance;
                p1 = rotatedCenters[i];
                p2 = rotatedCenters[j];
            }
        }
    }

    // qr midpoint
    Point2f midPoint = (p1 + p2) / 2.0;

    // Adjustment based on hypotenuse length to get qr outter corners
    double fullDiagonalLength = hypotenuseLength * (47.0 / 41.0);

    // length of square side
    double sideLength = fullDiagonalLength / sqrt(2);

    // bounding box for cropping
    int x = max(int(midPoint.x - sideLength / 2), 0);
    int y = max(int(midPoint.y - sideLength / 2), 0);
    int width = min(int(sideLength), rotatedImgColor.cols - x);
    int height = min(int(sideLength), rotatedImgColor.rows - y);

    // Crop & resize
    Mat croppedImg = rotatedImgColor(Rect(x, y, width, height));
    Mat scaledImg;
    resize(croppedImg, scaledImg, Size(940, 940), 0, 0, INTER_LINEAR);

    return scaledImg;
}

//////////////////////////////////////////////////////////////////////////
// Getting the block's color is based on simple Otsu's threshold for image histogram (test on heavy histogram distortions!)
// Compares each block pixel color (coverted into greyscale) against Otsu threshold to clamp it into the binary B/W predominant color
// Then calculates the most frequent/mode color based on the binary classification
//
// It looks to work rather well, even compensating for the qr distortions from rotation & scaling, contrast, brightness, noise and minor color variations

// get mode pixel color within block, accounting for some noise & lighting distortions
Vec3b blockColor(const Mat& img, const Point& center, int blockSize, double OtsuTresh) {
    std::map<Vec3b, int, decltype(&compareVec3b)> colorFrequency(&compareVec3b);
    int halfBlockSize = blockSize / 2;
    int startX = max(center.x - halfBlockSize, 0);
    int startY = max(center.y - halfBlockSize, 0);
    int endX = min(center.x + halfBlockSize, img.cols);
    int endY = min(center.y + halfBlockSize, img.rows);
    
    // iterate through each block pixel, then clamp its color
    for (int y = startY; y < endY; ++y) {
        for (int x = startX; x < endX; ++x) {
            Vec3b originalColor = img.at<Vec3b>(Point(x, y));
            Vec3b clampedColor = clampColor(originalColor, OtsuTresh);
            colorFrequency[clampedColor]++;
        }
    }
    Vec3b modeColor = {0, 0, 0};
    int maxCount = 0;
    for (const auto& kv : colorFrequency) {
        if (kv.second > maxCount) {
            maxCount = kv.second;
            modeColor = kv.first;
        }
    }
    return modeColor;
}

bool compareVec3b(const Vec3b &a, const Vec3b &b) {
    return a[0] < b[0] || (a[0] == b[0] && (a[1] < b[1] || (a[1] == b[1] && a[2] < b[2])));
}

// Otsu histogram threshold / おつさんありがとう！
double OtsuThresh(const Mat& src) {
    Mat grey, thresh;
    cvtColor(src, grey, COLOR_BGR2GRAY);
    double Otsu = threshold(grey, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
    return Otsu;
}

// clamp color based on Otsu threshhold
Vec3b clampColor(const Vec3b& color, double OtsuTresh) {
    Vec3b clampedColor;
    for (int i = 0; i < 3; i++) {
        clampedColor[i] = (color[i] >= OtsuTresh) ? 255 : 0;
    }
    return clampedColor;
}

// Color codes -> binary
int colorToBinary(const Vec3b& color) {
    int R = (color[2] == 255) ? 4 : 0;
    int G = (color[1] == 255) ? 2 : 0;
    int B = (color[0] == 255) ? 1 : 0;
    return R | G | B;
}

// Map binary to char
char binaryToChar(int binary) {
    if(binary >= 0 && binary < 64) {
        return encodingarray[binary];
    } else {
        cerr << "Invalid binary value!" << endl;
        return '?'; // if unknown -> ?
    }
}

//////////////////////////////////////////////////////////////////////////
// Decoding qr message
// Extract the useful area of the qr for decoding
// Creates a map of each grid block with id & binary color
// Based on mapped id, pairs blocks to decode character and then decoding the whole text

std::string decodeQR(const Mat& croppedScaledQR, double OtsuTresh) {
    const int gridSize = 47;
    int blockSize = croppedScaledQR.rows / gridSize;
    std::map<int, int> blockMap; // map block ID to its binary value
    std::string decodedText = ""; 

    int blockID = 1;

    // block area to decoded
    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            bool isInDecodingArea = (row < 6 && col >= 6 && col < 47) ||
                                    (row >= 6 && row < 41) ||
                                    (row >= 41 && row < 46 && col >= 6 && col < 41) ||
                                    (row == 46 && col >= 6 && col < 40);

            if (isInDecodingArea) {
                // Get color and convert to binary
                Vec3b color = blockColor(croppedScaledQR, Point(col * blockSize + blockSize / 2, row * blockSize + blockSize / 2), blockSize, OtsuTresh);
                int binaryValue = colorToBinary(color);
                blockMap[blockID++] = binaryValue;
            }
        }
    }
    // letter from block pairs
    for (int i = 1; i < blockID; i += 2) {
        int binaryValue1 = blockMap[i];
        int binaryValue2 = blockMap[i + 1];
        int combinedBinaryValue = (binaryValue1 << 3) | binaryValue2;
        decodedText += binaryToChar(combinedBinaryValue);
    }
    return decodedText;
}

//////////////////////////////////////////////////////////////////////////
// main
int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <Image_Path>" << endl;
        return -1;
    }

    Mat srcImg = imread(argv[1], IMREAD_COLOR);
    if (srcImg.empty()) {
        cerr << "Cannot load image!" << endl;
        return -1;
    }
    
    Mat processedImg;
    if (srcImg.channels() == 1) {
        cvtColor(srcImg, processedImg, COLOR_GRAY2BGR);
    } else {
        processedImg = srcImg.clone();
    }
   
    Mat hsvImg;
    cvtColor(processedImg, hsvImg, COLOR_BGR2HSV);

    auto [hsvCircles, greyCircles] = detectCircles(processedImg);
    vector<Point2f> greyCenters = getCircleCenters(greyCircles);
    vector<Point2f> hsvCenters = getCircleCenters(hsvCircles);
    vector<Point2f> centers;

    if (greyCenters.size() >= 3) {
        centers = greyCenters;
    } else if ((greyCenters.size() > 1 && greyCenters.size() < 3) && hsvCenters.size() >= 3) {
        centers = avgCenters(greyCircles, hsvCircles);
    } else if (hsvCenters.size() >= 3) {
        centers = hsvCenters;
    } else {
        cerr << "Error: Not enough circles detected!" << endl;
        return -1;
    }

    auto [rotatedImgColor, unusedImg, rotatedCenters, unusedPoints] = rotateQR(processedImg, srcImg, centers);
    if (rotatedImgColor.empty()) {
        cerr << "Rotation failed!" << endl;
        return -1;
    }

    Mat croppedScaledQR = cropScaleQR(rotatedImgColor, rotatedCenters);
    if (croppedScaledQR.empty()) {
        cerr << "Crop & scale failed!" << endl;
        return -1;
    }

    double Otsu = OtsuThresh(croppedScaledQR);
    string decodedText = decodeQR(croppedScaledQR, Otsu);
    cout << "Decoded text: \n" << decodedText << endl;

    return 0;
}