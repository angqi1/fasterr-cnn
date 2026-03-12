#ifndef FASTER_RCNN_DETECTOR_HPP
#define FASTER_RCNN_DETECTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

class FasterRCNNDetector {
public:
    FasterRCNNDetector(const std::string& engine_path, const std::string& labels_path,
                       int input_h = 800, int input_w = 1344);
    ~FasterRCNNDetector();

    std::vector<Detection> infer(const cv::Mat& image, float threshold = 0.5f);
    const std::vector<std::string>& getClassNames() const { return class_names_; }

private:
    void loadEngine(const std::string& engine_path);
    cv::Mat preprocess(const cv::Mat& image);

    // TensorRT 8.5+ 使用智能指针自动管理
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    void* d_input_ = nullptr;
    void* d_boxes_ = nullptr;
    void* d_scores_ = nullptr;
    void* d_labels_ = nullptr;

    float* h_boxes_ = nullptr;
    float* h_scores_ = nullptr;
    int* h_labels_ = nullptr;

    int input_h_, input_w_;
    std::vector<std::string> class_names_;
    size_t max_detections_ = 100;
};

#endif