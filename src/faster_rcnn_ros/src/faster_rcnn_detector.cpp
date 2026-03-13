#include "faster_rcnn_ros/faster_rcnn_detector.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInferPlugin.h>
#include <opencv2/dnn.hpp>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") + \
                std::to_string(__LINE__) + " — " + cudaGetErrorString(_err)); \
        } \
    } while(0)

// ─────────────────────────── TensorRT Logger ────────────────────────────────
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << "\n";
        }
    }
};

static TRTLogger gLogger;

// ─────────────────────────── Constructor ────────────────────────────────────
FasterRCNNDetector::FasterRCNNDetector(const std::string& engine_path,
                                       const std::string& labels_path,
                                       int input_h, int input_w)
    : input_h_(input_h), input_w_(input_w)
{
    // 加载类别标签
    std::ifstream label_file(labels_path);
    if (!label_file.good()) {
        throw std::runtime_error("Labels file not found: " + labels_path);
    }
    std::string line;
    while (std::getline(label_file, line)) {
        if (!line.empty()) class_names_.push_back(line);
    }

    // 创建 CUDA 流
    CHECK_CUDA(cudaStreamCreate(&stream_));

    // 加载 TRT engine
    loadEngine(engine_path);

    // 仅为输入分配固定显存（1×3×H×W float）
    CHECK_CUDA(cudaMalloc(&d_input_, 3 * input_h_ * input_w_ * sizeof(float)));

    // 动态输出分配器（DDS, 每个输出形状在推理时确定）
    scores_alloc_ = std::make_unique<OutputAllocator>();
    labels_alloc_ = std::make_unique<OutputAllocator>();
    boxes_alloc_  = std::make_unique<OutputAllocator>();

    // 设置输入形状与地址（固定分辨率，只需设一次）
    context_->setInputShape("image", nvinfer1::Dims4{1, 3, input_h_, input_w_});
    context_->setTensorAddress("image", d_input_);

    // 注册输出分配器
    context_->setOutputAllocator("scores", scores_alloc_.get());
    context_->setOutputAllocator("labels", labels_alloc_.get());
    context_->setOutputAllocator("boxes",  boxes_alloc_.get());
}

// ─────────────────────────── Destructor ─────────────────────────────────────
FasterRCNNDetector::~FasterRCNNDetector() {
    if (d_input_) cudaFree(d_input_);
    if (stream_)  cudaStreamDestroy(stream_);
    // OutputAllocator 析构函数自动释放各自的显存
}

// ─────────────────────────── Load Engine ────────────────────────────────────
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

    // 注册内置插件（必须在 deserialize 之前）
    initLibNvInferPlugins(&gLogger, "");

    std::unique_ptr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(gLogger)
    };
    if (!runtime) throw std::runtime_error("Failed to create TensorRT runtime");

    engine_.reset(runtime->deserializeCudaEngine(buffer.data(), size));
    if (!engine_) throw std::runtime_error("Failed to deserialize TRT engine");

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create execution context");
}

// ─────────────────────────── Inference ──────────────────────────────────────

// ── inferAsync: 预处理 + H2D + enqueueV3，不等待 GPU 完成 ─────────────────
void FasterRCNNDetector::inferAsync(const cv::Mat& image) {
    cv::dnn::blobFromImage(image, blob_,
                           1.0 / 255.0,
                           cv::Size(input_w_, input_h_),
                           cv::Scalar(),
                           /*swapRB=*/true,
                           /*crop=*/false,
                           CV_32F);

    const size_t input_bytes = 3 * input_h_ * input_w_ * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(d_input_, blob_.data, input_bytes,
                               cudaMemcpyHostToDevice, stream_));

    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    // ⚠️ 不 cudaStreamSynchronize：让调用方统一控制同步时机，实现并行
}

// ── syncAndCollect: 等待 GPU 完成，D2H 拷贝，后处理 ───────────────────────
std::vector<Detection> FasterRCNNDetector::syncAndCollect(const cv::Mat& image, float threshold) {
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    // ── 读取动态输出维度 ─────────────────────────────────────────────────
    int n_dets = 0;
    if (scores_alloc_->outputDims.nbDims > 0) {
        n_dets = static_cast<int>(scores_alloc_->outputDims.d[0]);
    }
    if (n_dets <= 0 || !scores_alloc_->buffer) {
        return {};
    }

    // ── 拷贝结果到主机 ───────────────────────────────────────────────────
    std::vector<float> h_scores(n_dets);
    std::vector<int>   h_labels(n_dets);
    std::vector<float> h_boxes(n_dets * 4);

    CHECK_CUDA(cudaMemcpy(h_scores.data(), scores_alloc_->buffer,
                          n_dets * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_labels.data(), labels_alloc_->buffer,
                          n_dets * sizeof(int),   cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_boxes.data(),  boxes_alloc_->buffer,
                          n_dets * 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // ── 坐标缩放：从 input 分辨率映射回原图 ─────────────────────────────
    const float sx = static_cast<float>(image.cols) / input_w_;
    const float sy = static_cast<float>(image.rows) / input_h_;

    std::vector<Detection> detections;
    detections.reserve(n_dets);

    for (int i = 0; i < n_dets; ++i) {
        if (h_scores[i] < threshold) continue;

        int x1 = std::max(0, std::min(static_cast<int>(h_boxes[i*4+0] * sx), image.cols - 1));
        int y1 = std::max(0, std::min(static_cast<int>(h_boxes[i*4+1] * sy), image.rows - 1));
        int x2 = std::max(0, std::min(static_cast<int>(h_boxes[i*4+2] * sx), image.cols - 1));
        int y2 = std::max(0, std::min(static_cast<int>(h_boxes[i*4+3] * sy), image.rows - 1));

        if (x2 <= x1 || y2 <= y1) continue;

        detections.push_back({
            cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
            h_scores[i],
            h_labels[i]
        });
    }
    return detections;
}

// ── infer: 同步接口，内部调用 inferAsync + syncAndCollect ─────────────────
std::vector<Detection> FasterRCNNDetector::infer(const cv::Mat& image, float threshold) {
    inferAsync(image);
    return syncAndCollect(image, threshold);
}
