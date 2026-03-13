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

// TRT 8.5+ IOutputAllocator: 动态形状输出，推理时由 TRT 回调分配显存
class OutputAllocator : public nvinfer1::IOutputAllocator {
public:
    ~OutputAllocator() {
        if (buffer) cudaFree(buffer);
    }

    // 仅在需要更大空间时重新分配，避免每帧 malloc/free
    void* reallocateOutput(char const* /*tensorName*/, void* /*currentMemory*/,
                           uint64_t size, uint64_t /*alignment*/) noexcept override {
        if (size > allocatedSize) {
            if (buffer) cudaFree(buffer);
            buffer = nullptr;
            if (cudaMalloc(&buffer, size) != cudaSuccess) return nullptr;
            allocatedSize = size;
        }
        return buffer;
    }

    // TRT 推理后回调，通知实际输出维度
    void notifyShape(char const* /*tensorName*/,
                     nvinfer1::Dims const& dims) noexcept override {
        outputDims = dims;
    }

    void*            buffer        = nullptr;
    uint64_t         allocatedSize = 0;
    nvinfer1::Dims   outputDims{};
};

class FasterRCNNDetector {
public:
    // engine: faster_rcnn.engine (FP16, input 1×3×375×1242)
    FasterRCNNDetector(const std::string& engine_path, const std::string& labels_path,
                       int input_h = 375, int input_w = 1242);
    ~FasterRCNNDetector();

    std::vector<Detection> infer(const cv::Mat& image, float threshold = 0.5f);
    const std::vector<std::string>& getClassNames() const { return class_names_; }

private:
    void loadEngine(const std::string& engine_path);

    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>  context_;

    void*        d_input_ = nullptr;   // 输入显存 (1×3×H×W float)
    cudaStream_t stream_  = nullptr;

    // 动态形状输出分配器（DDS）
    std::unique_ptr<OutputAllocator> scores_alloc_;
    std::unique_ptr<OutputAllocator> labels_alloc_;
    std::unique_ptr<OutputAllocator> boxes_alloc_;

    int input_h_, input_w_;
    std::vector<std::string> class_names_;
};

#endif