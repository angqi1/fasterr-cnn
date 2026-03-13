#ifndef FASTER_RCNN_DETECTOR_HPP
#define FASTER_RCNN_DETECTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

// YOLOv8 TRT Detector (COCO 80类, 固定输入 640×640, 输出 [1,84,8400])
class FasterRCNNDetector {
public:
    FasterRCNNDetector(const std::string& engine_path, const std::string& labels_path,
                       int input_h = 640, int input_w = 640);
    ~FasterRCNNDetector();

    std::vector<Detection> infer(const cv::Mat& image, float threshold = 0.5f);
    const std::vector<std::string>& getClassNames() const { return class_names_; }

private:
    void loadEngine(const std::string& engine_path);

    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>  context_;

    void*        d_input_  = nullptr;  // 输入显存  (1×3×640×640 float)
    void*        d_output_ = nullptr;  // 输出显存  (1×84×8400  float)
    cudaStream_t stream_   = nullptr;

    int input_h_, input_w_;

    // YOLOv8 固定超参
    static constexpr int   NUM_ANCHORS  = 8400;
    static constexpr int   NUM_CLASSES  = 80;
    static constexpr float NMS_THRESH   = 0.45f;

    std::vector<std::string> class_names_;
};

#endif