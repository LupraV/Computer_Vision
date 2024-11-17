// Luis Vieira 23012096
// GTSRB: Comparing Performance of Preprocessing Techniques for Traffic Sign Recognition Using a HOG-SVM

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <string>
#include <algorithm>
#include <random>
#include <map>
#include <memory>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <getopt.h>

void setSeed() {
    std::srand(123);
}

struct Annotation {
    std::string filename;
    int width;
    int height;
    int x1, y1, x2, y2;
    int classId;
};

// assigning a type of road sign
//const std::vector<unsigned int> Prohibitory = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16};
//const std::vector<unsigned int> Mandatory = {33, 34, 35, 36, 37, 38, 39, 40};
//const std::vector<unsigned int> Danger = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
//const std::vector<unsigned int> ALL = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 33, 34, 35, 36, 37, 38, 39, 40, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

// get annotations/metadata from img - ROI, class
std::vector<Annotation> loadAnnotations(const std::string& csvFile, char delimiter = ';') { // accounting for the annoying ";" instead of comma
    std::vector<Annotation> annotations;
    std::ifstream file(csvFile);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << csvFile << std::endl;
        return annotations;
    }

    std::string line;
    std::getline(file, line); // skipping header
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        Annotation annotation;
        std::string token;
        std::getline(lineStream, annotation.filename, delimiter);
        std::getline(lineStream, token, delimiter); annotation.width = std::stoi(token);
        std::getline(lineStream, token, delimiter); annotation.height = std::stoi(token);
        std::getline(lineStream, token, delimiter); annotation.x1 = std::stoi(token);
        std::getline(lineStream, token, delimiter); annotation.y1 = std::stoi(token);
        std::getline(lineStream, token, delimiter); annotation.x2 = std::stoi(token);
        std::getline(lineStream, token, delimiter); annotation.y2 = std::stoi(token);
        std::getline(lineStream, token, delimiter); annotation.classId = std::stoi(token);
        annotations.push_back(annotation);
    }
    return annotations;
}

// shuffle dataset on split (minimise bias, imbalance, overfitting) seed=123
void splitDataset(const std::vector<Annotation>& annotations, std::vector<Annotation>& trainSet, std::vector<Annotation>& valSet, float trainRatio) {
    std::vector<Annotation> shuffled = annotations;
    std::mt19937 g(123);
    std::shuffle(shuffled.begin(), shuffled.end(), g);

    size_t trainSize = static_cast<size_t>(trainRatio * annotations.size());
    trainSet.assign(shuffled.begin(), shuffled.begin() + trainSize);
    valSet.assign(shuffled.begin() + trainSize, shuffled.end());
}

cv::Mat loadImage(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error loading image " << path << std::endl;
    }
    return image;
}

cv::Mat resizeImage(const cv::Mat& image, int targetWidth, int targetHeight) {
    cv::Mat resizedImage;
    cv::resize(image, resizedImage, cv::Size(targetWidth, targetHeight));
    return resizedImage;
}

void applyCLAHE(cv::Mat& image) {
    cv::Mat lab_image;
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> lab_planes;
    cv::split(lab_image, lab_planes);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lab_planes[0], mean, stddev);
    // dynamic CLAHE based on the std dev
    double clipLimit;
    cv::Size tileGridSize(8, 8);
    if (stddev[0] < 50) {  // low contrast
        clipLimit = 4.0;
    } else if (stddev[0] < 100) {  // medium
        clipLimit = 2.0;
    } else {  // high 
        clipLimit = 1.0;
    }
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
    clahe->apply(lab_planes[0], lab_planes[0]);
    cv::merge(lab_planes, lab_image);
    cv::cvtColor(lab_image, image, cv::COLOR_Lab2BGR);
}

// HOG w/ Gaussian blur (window size, Block size, stride, cell size, bins )
cv::Mat computeHOG(const cv::Mat& image, int kernelSize = 3, double sigma = 0) {
    cv::Mat processedImage;
    cv::GaussianBlur(image, processedImage, cv::Size(kernelSize, kernelSize), sigma);
    cv::HOGDescriptor hog(cv::Size(32, 32), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
    std::vector<float> descriptors;
    hog.compute(processedImage, descriptors);
    return cv::Mat(descriptors).clone().reshape(1, 1);
}

// HUE
cv::Mat extractHUE(const cv::Mat& image) {
    cv::Mat hsv_image, hue_channel;
    cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_planes;
    cv::split(hsv_image, hsv_planes);
    cv::Mat hue_channel_equalized;
    cv::equalizeHist(hsv_planes[0], hue_channel_equalized);
    hsv_planes[0] = hue_channel_equalized;
    cv::merge(hsv_planes, hsv_image);
    cv::cvtColor(hsv_image, hue_channel, cv::COLOR_HSV2BGR);
    return hue_channel;
}

// SVM Classifier
int predictSVM(const cv::Ptr<cv::ml::SVM>& svm, const cv::Mat& features) {
    cv::Mat response;
    svm->predict(features, response);
    return static_cast<int>(response.at<float>(0, 0));
}

// get metrics, confusion matrix & misclassified images
void computeMetrics(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, const std::vector<std::string>& imagePaths, int numClasses, const std::string& outputPath) {
    cv::Mat confusionMatrix = cv::Mat::zeros(numClasses, numClasses, CV_32S);
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << outputPath << "." << std::endl;
        return;
    }
    std::vector<std::string> failedImages;
    std::vector<int> misclassifiedCount(numClasses, 0);

    for (size_t i = 0; i < trueLabels.size(); ++i) {
        confusionMatrix.at<int>(trueLabels[i], predictedLabels[i])++;
        if (trueLabels[i] != predictedLabels[i]) {
            failedImages.push_back(std::filesystem::path(imagePaths[i]).filename().string());
            misclassifiedCount[trueLabels[i]]++;
        }
    }
    file << "Confusion Matrix:\n" << confusionMatrix << std::endl;

    double accuracy = 0;
    double precision = 0;
    double recall = 0;
    double f1Score = 0;
    int truePositives = 0;
    int totalSamples = 0;
    std::vector<int> falsePositives(numClasses, 0);
    std::vector<int> falseNegatives(numClasses, 0);

    for (int i = 0; i < numClasses; ++i) {
        truePositives += confusionMatrix.at<int>(i, i);
        totalSamples += cv::sum(confusionMatrix.row(i))[0];

        for (int j = 0; j < numClasses; ++j) {
            if (i != j) {
                falsePositives[j] += confusionMatrix.at<int>(i, j);
                falseNegatives[i] += confusionMatrix.at<int>(i, j);
            }
        }
    }
    accuracy = static_cast<double>(truePositives) / totalSamples;

    for (int i = 0; i < numClasses; ++i) {
        double classPrecision = static_cast<double>(confusionMatrix.at<int>(i, i)) / (confusionMatrix.at<int>(i, i) + falsePositives[i]);
        double classRecall = static_cast<double>(confusionMatrix.at<int>(i, i)) / (confusionMatrix.at<int>(i, i) + falseNegatives[i]);

       if (!std::isnan(classPrecision) && !std::isnan(classRecall)) {
            precision += classPrecision;
            recall += classRecall;
            f1Score += 2 * (classPrecision * classRecall) / (classPrecision + classRecall);
        }
    }

    precision /= numClasses;
    recall /= numClasses;
    f1Score /= numClasses;
    file << "Accuracy: " << accuracy << "\n";
    file << "Precision: " << precision << "\n";
    file << "Recall: " << recall << "\n";
    file << "F1 Score: " << f1Score << "\n";
    file << "Misclassified Images by Class:\n";
    for (int i = 0; i < numClasses; ++i) {
        file << "Class " << i << ": " << misclassifiedCount[i] << "\n";
    }
    file << "Failed Images:\n";
    for (const auto& imageName : failedImages) {
        file << imageName << "\n";
    }
    file.close();
    std::cout << "F1 Score: " << f1Score << "\n";
    std::cout << "Accuracy: " << accuracy << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall: " << recall << "\n";
}

// progress bar
void printProgressBar(const std::string& title, int current, int total) {
    int barWidth = 80;
    float progress = static_cast<float>(current) / total;

    std::cout << title << ": [";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

void timerStart(std::chrono::steady_clock::time_point& start) {
    start = std::chrono::steady_clock::now();
}

// SVM Train
cv::Ptr<cv::ml::SVM> trainSVM(const cv::Mat& trainData, const cv::Mat& trainLabels, double C, double gamma) {
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setC(C);
    svm->setGamma(gamma);
    svm->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);
    return svm;
}

void timerStop(const std::string& phase, const std::chrono::steady_clock::time_point& start) {
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    int minutes = duration.count() / 60;
    int seconds = duration.count() % 60;
    std::cout << phase << " completed in " << minutes << " min " << seconds << " sec.\n";
}

int main(int argc, char** argv) {
    setSeed();
    int runHOG = 0, runCLAHEHOG = 0, runYUVHOG = 0, runHUEHOG = 0, runCLAHEYUVHOG = 0, runHUEYUVHOG = 0, runCLAHEHUEYUVHOG = 0;

    static struct option long_options[] = {
        {"hog", no_argument, &runHOG, 1},
        {"clahehog", no_argument, &runCLAHEHOG, 1},
        {"yuvhog", no_argument, &runYUVHOG, 1},
        {"huehog", no_argument, &runHUEHOG, 1},
        {"claheyuvhog", no_argument, &runCLAHEYUVHOG, 1},
        {"hueyuvhog", no_argument, &runHUEYUVHOG, 1},
        {"clahehueyuvhog", no_argument, &runCLAHEHUEYUVHOG, 1},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    while (getopt_long(argc, argv, "", long_options, &option_index) != -1);

    // paths
    std::string basePath = std::filesystem::current_path().parent_path().string() + "/";
    std::string trainImagesPath = basePath + "data/Final_Training/Images";
    std::string testImagesPath = basePath + "data/Final_Test/Images";
    std::string testAnnotationsPath = basePath + "data/Final_Test/Images/GT-final_test.csv";

    // loading and preprocess images
    std::vector<Annotation> annotations;
    for (int i = 0; i < 43; ++i) {
        std::string classStr = std::to_string(i);
        std::string classDir = trainImagesPath + "/" + std::string(5 - classStr.length(), '0') + classStr;
        std::string csvFile = classDir + "/GT-" + std::string(5 - classStr.length(), '0') + classStr + ".csv";
        std::filesystem::path absoluteCsvPath = std::filesystem::absolute(csvFile);
        if (!std::filesystem::exists(csvFile)) {
            std::cerr << "Error: File does not exist " << absoluteCsvPath << std::endl;
            continue;
        }
        std::vector<Annotation> classAnnotations = loadAnnotations(csvFile);
        for (auto& annotation : classAnnotations) {
            annotation.filename = classDir + "/" + annotation.filename;
        }
        annotations.insert(annotations.end(), classAnnotations.begin(), classAnnotations.end());
    }
    if (annotations.empty()) {
        std::cerr << "Error: No training data available." << std::endl;
        return -1;
    }

    // Split dataset 80:20 train:validation
    std::vector<Annotation> trainSet, valSet;
    splitDataset(annotations, trainSet, valSet, 0.8);

    // Train SVM models
    std::vector<cv::Mat> trainFeaturesHOG, trainFeaturesCLAHEHOG, trainFeaturesYUVHOG, trainFeaturesHUEHOG, trainFeaturesCLAHEYUVHOG, trainFeaturesHUEYUVHOG, trainFeaturesCLAHEHUEYUVHOG;
    std::vector<int> trainLabels;

    int totalFiles = trainSet.size();
    std::chrono::steady_clock::time_point start;
    timerStart(start);
    for (int i = 0; i < totalFiles; ++i) {
        const auto& annotation = trainSet[i];
        std::string imagePath = annotation.filename;
        cv::Mat image = loadImage(imagePath);
        if (image.empty()) {
            continue;
        }

        // Preprocessing
        cv::Mat resizedImage = resizeImage(image, 32, 32);

        if (runHOG) {
            trainFeaturesHOG.push_back(computeHOG(resizedImage));
        }
        if (runCLAHEHOG) {
            applyCLAHE(resizedImage);
            trainFeaturesCLAHEHOG.push_back(computeHOG(resizedImage));
        }
        if (runYUVHOG) {
            cv::Mat yuv_image;
            cv::cvtColor(resizedImage, yuv_image, cv::COLOR_BGR2YUV);
            trainFeaturesYUVHOG.push_back(computeHOG(yuv_image));
        }
        if (runHUEHOG) {
            cv::Mat hue_image = extractHUE(resizedImage);
            trainFeaturesHUEHOG.push_back(computeHOG(hue_image));
        }
        if (runCLAHEYUVHOG) {
            applyCLAHE(resizedImage);
            cv::Mat yuv_image;
            cv::cvtColor(resizedImage, yuv_image, cv::COLOR_BGR2YUV);
            trainFeaturesCLAHEYUVHOG.push_back(computeHOG(yuv_image));
        }
        if (runHUEYUVHOG) {
            cv::Mat hue_image = extractHUE(resizedImage);
            cv::Mat yuv_image;
            cv::cvtColor(hue_image, yuv_image, cv::COLOR_BGR2YUV);
            trainFeaturesHUEYUVHOG.push_back(computeHOG(yuv_image));
        }
        if (runCLAHEHUEYUVHOG) {
            applyCLAHE(resizedImage);
            cv::Mat hue_image = extractHUE(resizedImage);
            cv::Mat yuv_image;
            cv::cvtColor(hue_image, yuv_image, cv::COLOR_BGR2YUV);
            trainFeaturesCLAHEHUEYUVHOG.push_back(computeHOG(yuv_image));
        }
        trainLabels.push_back(annotation.classId);

        printProgressBar("Training feature extraction", i + 1, totalFiles);
    }
    timerStop("Training feature extraction", start);

    std::cout << std::endl;

    if (trainFeaturesHOG.empty() && trainFeaturesCLAHEHOG.empty() && trainFeaturesYUVHOG.empty() && trainFeaturesHUEHOG.empty() && trainFeaturesCLAHEYUVHOG.empty() && trainFeaturesHUEYUVHOG.empty() && trainFeaturesCLAHEHUEYUVHOG.empty()) {
        std::cerr << "Error: No training data available after preprocessing." << std::endl;
        return -1;
    }

    cv::Mat trainDataHOG, trainDataCLAHEHOG, trainDataYUVHOG, trainDataHUEHOG, trainDataCLAHEYUVHOG, trainDataHUEYUVHOG, trainDataCLAHEHUEYUVHOG;
    if (runHOG) {
        cv::vconcat(trainFeaturesHOG, trainDataHOG);
    }
    if (runCLAHEHOG) {
        cv::vconcat(trainFeaturesCLAHEHOG, trainDataCLAHEHOG);
    }
        if (runYUVHOG) {
        cv::vconcat(trainFeaturesYUVHOG, trainDataYUVHOG);
    }
    if (runHUEHOG) {
        cv::vconcat(trainFeaturesHUEHOG, trainDataHUEHOG);
    }
    if (runCLAHEYUVHOG) {
        cv::vconcat(trainFeaturesCLAHEYUVHOG, trainDataCLAHEYUVHOG);
    }
    if (runHUEYUVHOG) {
        cv::vconcat(trainFeaturesHUEYUVHOG, trainDataHUEYUVHOG);
    }
    if (runCLAHEHUEYUVHOG) {
        cv::vconcat(trainFeaturesCLAHEHUEYUVHOG, trainDataCLAHEHUEYUVHOG);
    }

    cv::Mat trainLabelsMat(trainLabels, true);
    trainLabelsMat.convertTo(trainLabelsMat, CV_32S);

    cv::Ptr<cv::ml::SVM> svmHOG, svmCLAHEHOG, svmYUVHOG, svmHUEHOG, svmCLAHEYUVHOG, svmHUEYUVHOG, svmCLAHEHUEYUVHOG;
    // RandomizedSearchCV 3-fold 10 iteration ('C':(5, 25, 10),'gamma': (0.05, 0.35, 10), ran in python wasn't working correctly here!)
    // best param: 'C': 9.4445, 11.6667, *20.5557, 22.7778;  'gamma': *0.2167, 0.2833, 0.3166, 0.35
    double C = 20.5557; 
    double gamma = 0.2167;
    if (runHOG) {
        svmHOG = trainSVM(trainDataHOG, trainLabelsMat, C, gamma);
        svmHOG->save(basePath + "output/svm_hog_model.xml");
    }
    if (runCLAHEHOG) {
        svmCLAHEHOG = trainSVM(trainDataCLAHEHOG, trainLabelsMat, C, gamma);
        svmCLAHEHOG->save(basePath + "output/svm_clahe_hog_model.xml");
    }
    if (runYUVHOG) {
        svmYUVHOG = trainSVM(trainDataYUVHOG, trainLabelsMat, C, gamma);
        svmYUVHOG->save(basePath + "output/svm_yuv_hog_model.xml");
    }
    if (runHUEHOG) {
        svmHUEHOG = trainSVM(trainDataHUEHOG, trainLabelsMat, C, gamma);
        svmHUEHOG->save(basePath + "output/svm_hue_hog_model.xml");
    }
    if (runCLAHEYUVHOG) {
        svmCLAHEYUVHOG = trainSVM(trainDataCLAHEYUVHOG, trainLabelsMat, C, gamma);
        svmCLAHEYUVHOG->save(basePath + "output/svm_claheyuv_hog_model.xml");
    }
    if (runHUEYUVHOG) {
        svmHUEYUVHOG = trainSVM(trainDataHUEYUVHOG, trainLabelsMat, C, gamma);
        svmHUEYUVHOG->save(basePath + "output/svm_hueyuv_hog_model.xml");
    }
    if (runCLAHEHUEYUVHOG) {
        svmCLAHEHUEYUVHOG = trainSVM(trainDataCLAHEHUEYUVHOG, trainLabelsMat, C, gamma);
        svmCLAHEHUEYUVHOG->save(basePath + "output/svm_clahehueyuv_hog_model.xml");
    }

    std::vector<int> valTrueLabels;
    std::vector<int> valPredictedLabelsHOG, valPredictedLabelsCLAHEHOG, valPredictedLabelsYUVHOG, valPredictedLabelsHUEHOG, valPredictedLabelsCLAHEYUVHOG, valPredictedLabelsHUEYUVHOG, valPredictedLabelsCLAHEHUEYUVHOG;
    std::vector<std::string> valImagePaths;

    totalFiles = valSet.size();
    timerStart(start);
    for (int i = 0; i < totalFiles; ++i) {
        const auto& annotation = valSet[i];
        std::string imagePath = annotation.filename;
        cv::Mat image = loadImage(imagePath);
        if (image.empty()) {
            continue;
        }
        cv::Mat resizedImage = resizeImage(image, 32, 32);

        if (runHOG) {
            valPredictedLabelsHOG.push_back(predictSVM(svmHOG, computeHOG(resizedImage)));
        }
        if (runCLAHEHOG) {
            applyCLAHE(resizedImage);
            valPredictedLabelsCLAHEHOG.push_back(predictSVM(svmCLAHEHOG, computeHOG(resizedImage)));
        }
        if (runYUVHOG) {
            cv::Mat yuv_image;
            cv::cvtColor(resizedImage, yuv_image, cv::COLOR_BGR2YUV);
            valPredictedLabelsYUVHOG.push_back(predictSVM(svmYUVHOG, computeHOG(yuv_image)));
        }
        if (runHUEHOG) {
            cv::Mat hue_image = extractHUE(resizedImage);
            valPredictedLabelsHUEHOG.push_back(predictSVM(svmHUEHOG, computeHOG(hue_image)));
        }
        if (runCLAHEYUVHOG) {
            applyCLAHE(resizedImage);
            cv::Mat yuv_image;
            cv::cvtColor(resizedImage, yuv_image, cv::COLOR_BGR2YUV);
            valPredictedLabelsCLAHEYUVHOG.push_back(predictSVM(svmCLAHEYUVHOG, computeHOG(yuv_image)));
        }
        if (runHUEYUVHOG) {
            cv::Mat hue_image = extractHUE(resizedImage);
            cv::Mat yuv_image;
            cv::cvtColor(hue_image, yuv_image, cv::COLOR_BGR2YUV);
            valPredictedLabelsHUEYUVHOG.push_back(predictSVM(svmHUEYUVHOG, computeHOG(yuv_image)));
        }
        if (runCLAHEHUEYUVHOG) {
        applyCLAHE(resizedImage);
        cv::Mat hue_image = extractHUE(resizedImage);
        cv::Mat yuv_image;
        cv::cvtColor(hue_image, yuv_image, cv::COLOR_BGR2YUV);
        valPredictedLabelsCLAHEHUEYUVHOG.push_back(predictSVM(svmCLAHEHUEYUVHOG, computeHOG(yuv_image)));
    }
        valTrueLabels.push_back(annotation.classId);
        valImagePaths.push_back(imagePath);

        printProgressBar("Validation feature extraction", i + 1, totalFiles);
    }
    timerStop("Validation feature extraction", start);

    std::cout << std::endl;

    if (runHOG) {
        computeMetrics(valTrueLabels, valPredictedLabelsHOG, valImagePaths, 43, basePath + "output/evaluation_results_val_hog.txt");
    }
    if (runCLAHEHOG) {
        computeMetrics(valTrueLabels, valPredictedLabelsCLAHEHOG, valImagePaths, 43, basePath + "output/evaluation_results_val_clahe_hog.txt");
    }
    if (runYUVHOG) {
        computeMetrics(valTrueLabels, valPredictedLabelsYUVHOG, valImagePaths, 43, basePath + "output/evaluation_results_val_yuv_hog.txt");
    }
    if (runHUEHOG) {
        computeMetrics(valTrueLabels, valPredictedLabelsHUEHOG, valImagePaths, 43, basePath + "output/evaluation_results_val_hue_hog.txt");
    }
    if (runCLAHEYUVHOG) {
        computeMetrics(valTrueLabels, valPredictedLabelsCLAHEYUVHOG, valImagePaths, 43, basePath + "output/evaluation_results_val_claheyuv_hog.txt");
    }
    if (runHUEYUVHOG) {
        computeMetrics(valTrueLabels, valPredictedLabelsHUEYUVHOG, valImagePaths, 43, basePath + "output/evaluation_results_val_hueyuv_hog.txt");
    }
    if (runCLAHEHUEYUVHOG) {
        computeMetrics(valTrueLabels, valPredictedLabelsCLAHEHUEYUVHOG, valImagePaths, 43, basePath + "output/evaluation_results_val_clahehueyuv_hog.txt");
    }

    std::vector<int> testTrueLabels;
    std::vector<int> testPredictedLabelsHOG, testPredictedLabelsCLAHEHOG, testPredictedLabelsYUVHOG, testPredictedLabelsHUEHOG, testPredictedLabelsCLAHEYUVHOG, testPredictedLabelsHUEYUVHOG, testPredictedLabelsCLAHEHUEYUVHOG;
    std::vector<std::string> testImagePaths;

    std::vector<Annotation> testAnnotations = loadAnnotations(testAnnotationsPath);
    totalFiles = testAnnotations.size();
    timerStart(start);
    for (int i = 0; i < totalFiles; ++i) {
        const auto& annotation = testAnnotations[i];
        std::string imagePath = testImagesPath + "/" + annotation.filename;
        cv::Mat image = loadImage(imagePath);
        if (image.empty()) {
            continue;
        }
        cv::Mat resizedImage = resizeImage(image, 32, 32);

        if (runHOG) {
            testPredictedLabelsHOG.push_back(predictSVM(svmHOG, computeHOG(resizedImage)));
        }
        if (runCLAHEHOG) {
            applyCLAHE(resizedImage);
            testPredictedLabelsCLAHEHOG.push_back(predictSVM(svmCLAHEHOG, computeHOG(resizedImage)));
        }
        if (runYUVHOG) {
            cv::Mat yuv_image;
            cv::cvtColor(resizedImage, yuv_image, cv::COLOR_BGR2YUV);
            testPredictedLabelsYUVHOG.push_back(predictSVM(svmYUVHOG, computeHOG(yuv_image)));
        }
        if (runHUEHOG) {
            cv::Mat hue_image = extractHUE(resizedImage);
            testPredictedLabelsHUEHOG.push_back(predictSVM(svmHUEHOG, computeHOG(hue_image)));
        }
        if (runCLAHEYUVHOG) {
            applyCLAHE(resizedImage);
            cv::Mat yuv_image;
            cv::cvtColor(resizedImage, yuv_image, cv::COLOR_BGR2YUV);
            testPredictedLabelsCLAHEYUVHOG.push_back(predictSVM(svmCLAHEYUVHOG, computeHOG(yuv_image)));
        }
        if (runHUEYUVHOG) {
            cv::Mat hue_image = extractHUE(resizedImage);
            cv::Mat yuv_image;
            cv::cvtColor(hue_image, yuv_image, cv::COLOR_BGR2YUV);
            testPredictedLabelsHUEYUVHOG.push_back(predictSVM(svmHUEYUVHOG, computeHOG(yuv_image)));
        }
        if (runCLAHEHUEYUVHOG) {
            applyCLAHE(resizedImage);
            cv::Mat hue_image = extractHUE(resizedImage);
            cv::Mat yuv_image;
            cv::cvtColor(hue_image, yuv_image, cv::COLOR_BGR2YUV);
            testPredictedLabelsCLAHEHUEYUVHOG.push_back(predictSVM(svmCLAHEHUEYUVHOG, computeHOG(yuv_image)));
        }
        testTrueLabels.push_back(annotation.classId);
        testImagePaths.push_back(imagePath);

        printProgressBar("Test feature extraction", i + 1, totalFiles);
    }
    timerStop("Test feature extraction", start);

    std::cout << std::endl;

    if (runHOG) {
        computeMetrics(testTrueLabels, testPredictedLabelsHOG, testImagePaths, 43, basePath + "output/evaluation_results_test_hog.txt");
    }
    if (runCLAHEHOG) {
        computeMetrics(testTrueLabels, testPredictedLabelsCLAHEHOG, testImagePaths, 43, basePath + "output/evaluation_results_test_clahe_hog.txt");
    }
    if (runYUVHOG) {
        computeMetrics(testTrueLabels, testPredictedLabelsYUVHOG, testImagePaths, 43, basePath + "output/evaluation_results_test_yuv_hog.txt");
    }
    if (runHUEHOG) {
        computeMetrics(testTrueLabels, testPredictedLabelsHUEHOG, testImagePaths, 43, basePath + "output/evaluation_results_test_hue_hog.txt");
    }
    if (runCLAHEYUVHOG) {
        computeMetrics(testTrueLabels, testPredictedLabelsCLAHEYUVHOG, testImagePaths, 43, basePath + "output/evaluation_results_test_claheyuv_hog.txt");
    }
    if (runHUEYUVHOG) {
        computeMetrics(testTrueLabels, testPredictedLabelsHUEYUVHOG, testImagePaths, 43, basePath + "output/evaluation_results_test_hueyuv_hog.txt");
    }
    if (runCLAHEHUEYUVHOG) {
        computeMetrics(testTrueLabels, testPredictedLabelsCLAHEHUEYUVHOG, testImagePaths, 43, basePath + "output/evaluation_results_test_clahehueyuv_hog.txt");
    }

    return 0;
}