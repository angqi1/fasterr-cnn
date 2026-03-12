#include "faster_rcnn_ros/faster_rcnn_detector.hpp"
#include <fstream>
#include <cuda_runtime.h>
#include <NvInferPlugin.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

// ✅ TensorRT 8.5+ 自定义 Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

FasterRCNNDetector::FasterRCNNDetector(const std::string& engine_path,
                                       const std::string& labels_path,
                                       int input_h, int input_w)
    : input_h_(input_h), input_w_(input_w) {
    
    std::ifstream label_file(labels_path);
    std::string line;
    while (std::getline(label_file, line)) {
        class_names_.push_back(line);
    }

    loadEngine(engine_path);

    // Allocate GPU buffers
    size_t input_size = 3 * input_h_ * input_w_ * sizeof(float);
    size_t boxes_size = max_detections_ * 4 * sizeof(float);
    size_t scores_size = max_detections_ * sizeof(float);
    size_t labels_size = max_detections_ * sizeof(int);

    CHECK_CUDA(cudaMalloc(&d_input_, input_size));
    CHECK_CUDA(cudaMalloc(&d_boxes_, boxes_size));
    CHECK_CUDA(cudaMalloc(&d_scores_, scores_size));
    CHECK_CUDA(cudaMalloc(&d_labels_, labels_size));

    h_boxes_ = new float[max_detections_ * 4];
    h_scores_ = new float[max_detections_];
    h_labels_ = new int[max_detections_];
}

FasterRCNNDetector::~FasterRCNNDetector() {
    if (d_input_) cudaFree(d_input_);
    if (d_boxes_) cudaFree(d_boxes_);
    if (d_scores_) cudaFree(d_scores_);
    if (d_labels_) cudaFree(d_labels_);
    delete[] h_boxes_;
    delete[] h_scores_;
    delete[] h_labels_;
}

void FasterRCNNDetector::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Engine file not found: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    static Logger logger; // 静态避免重复构造
    initLibNvInferPlugins(&logger, "");  // 注册所有内置TRT插件
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(buffer.data(), size)
    );
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize engine");
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext()
    );
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }
}

cv::Mat FasterRCNNDetector::preprocess(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_w_, input_h_));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32F, 1.0f / 255.0f);
    return resized;
}

std::vector<Detection> FasterRCNNDetector::infer(const cv::Mat& image, float threshold) {
    cv::Mat preprocessed = preprocess(image);
    std::vector<float> input_data(preprocessed.total() * preprocessed.channels());
    std::memcpy(input_data.data(), preprocessed.data, input_data.size() * sizeof(float));

    CHECK_CUDA(cudaMemcpy(d_input_, input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice));

    // ✅ TensorRT 8.5+ executeV2
    void* bindings[] = {d_input_, d_boxes_, d_scores_, d_labels_};
    bool status = context_->executeV2(bindings);
    if (!status) {
        throw std::runtime_error("TensorRT inference failed");
    }

    CHECK_CUDA(cudaMemcpy(h_boxes_, d_boxes_, max_detections_ * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_scores_, d_scores_, max_detections_ * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_labels_, d_labels_, max_detections_ * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<Detection> detections;
    for (size_t i = 0; i < max_detections_; ++i) {
        if (h_scores_[i] > threshold) {
            int x1 = static_cast<int>(h_boxes_[i * 4 + 0] * image.cols / input_w_);
            int y1 = static_cast<int>(h_boxes_[i * 4 + 1] * image.rows / input_h_);
            int x2 = static_cast<int>(h_boxes_[i * 4 + 2] * image.cols / input_w_);
            int y2 = static_cast<int>(h_boxes_[i * 4 + 3] * image.rows / input_h_);

            detections.push_back({
                cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
                h_scores_[i],
                h_labels_[i]
            });
        }
    }
    return detections;
}