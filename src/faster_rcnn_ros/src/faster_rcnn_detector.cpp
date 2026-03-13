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

    // 分配输入显存 (1×3×640×640)
    CHECK_CUDA(cudaMalloc(&d_input_,  3 * input_h_ * input_w_ * sizeof(float)));

    // 分配输出显存 (1×84×8400)
    CHECK_CUDA(cudaMalloc(&d_output_, (4 + NUM_CLASSES) * NUM_ANCHORS * sizeof(float)));

    // 绑定张量地址（YOLOv8 静态形状，无需 setInputShape）
    context_->setTensorAddress("images",  d_input_);
    context_->setTensorAddress("output0", d_output_);
}

// ─────────────────────────── Destructor ─────────────────────────────────────
FasterRCNNDetector::~FasterRCNNDetector() {
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_)   cudaStreamDestroy(stream_);
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
std::vector<Detection> FasterRCNNDetector::infer(const cv::Mat& image, float threshold) {
    // ── 1. Letterbox 预处理（YOLOv8 标准训练预处理）──────────────────────
    // scale = min(640/H, 640/W)，等比缩放 + 灰边填充，保持宽高比
    float scale = std::min(
        static_cast<float>(input_h_) / static_cast<float>(image.rows),
        static_cast<float>(input_w_) / static_cast<float>(image.cols)
    );
    int new_w = static_cast<int>(std::round(image.cols * scale));
    int new_h = static_cast<int>(std::round(image.rows * scale));
    int pad_x = (input_w_ - new_w) / 2;
    int pad_y = (input_h_ - new_h) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 处理 RGBA 图像（如 kace.png）
    if (resized.channels() == 4) cv::cvtColor(resized, resized, cv::COLOR_BGRA2BGR);

    cv::Mat canvas(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // ── 2. BGR→RGB + /255.0 + CHW 排列 ───────────────────────────────────
    cv::Mat blob;
    cv::dnn::blobFromImage(canvas, blob,
                           1.0 / 255.0,
                           cv::Size(input_w_, input_h_),
                           cv::Scalar(),
                           /*swapRB=*/true,
                           /*crop=*/false,
                           CV_32F);

    // ── 3. 上传输入到 GPU ─────────────────────────────────────────────────
    const size_t input_bytes = 3 * input_h_ * input_w_ * sizeof(float);
    CHECK_CUDA(cudaMemcpyAsync(d_input_, blob.data, input_bytes,
                               cudaMemcpyHostToDevice, stream_));

    // ── 4. TRT 推理 ───────────────────────────────────────────────────────
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    CHECK_CUDA(cudaStreamSynchronize(stream_));

    // ── 5. 下载输出 (1×84×8400) ───────────────────────────────────────────
    const int out_size = (4 + NUM_CLASSES) * NUM_ANCHORS;
    std::vector<float> h_output(out_size);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output_,
                          out_size * sizeof(float), cudaMemcpyDeviceToHost));

    // ── 6. 解码 YOLOv8 输出 [84, 8400] ────────────────────────────────────
    // 内存布局: h_output[c * NUM_ANCHORS + i] = anchor i 的 channel c
    //   [0:4]  = cx, cy, w, h  (640×640 像素空间)
    //   [4:84] = 80 类别分数（已经过 sigmoid）
    std::vector<cv::Rect>  boxes_nms;
    std::vector<float>     scores_nms;
    std::vector<int>       class_nms;

    for (int i = 0; i < NUM_ANCHORS; ++i) {
        // 找最大类别分数
        float max_score = 0.f;
        int   max_cls   = 0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float s = h_output[(4 + c) * NUM_ANCHORS + i];
            if (s > max_score) { max_score = s; max_cls = c; }
        }
        if (max_score < threshold) continue;

        float cx = h_output[0 * NUM_ANCHORS + i];
        float cy = h_output[1 * NUM_ANCHORS + i];
        float bw = h_output[2 * NUM_ANCHORS + i];
        float bh = h_output[3 * NUM_ANCHORS + i];

        int ix = static_cast<int>(cx - bw / 2.f);
        int iy = static_cast<int>(cy - bh / 2.f);
        int iw = static_cast<int>(bw);
        int ih = static_cast<int>(bh);
        boxes_nms.emplace_back(ix, iy, iw, ih);
        scores_nms.push_back(max_score);
        class_nms.push_back(max_cls);
    }

    if (boxes_nms.empty()) return {};

    // ── 7. NMS ────────────────────────────────────────────────────────────
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes_nms, scores_nms, threshold, NMS_THRESH, keep);

    // ── 8. 坐标反变换：letterbox 网络坐标 → 原图坐标 ─────────────────────
    std::vector<Detection> detections;
    detections.reserve(keep.size());

    for (int idx : keep) {
        const auto& b = boxes_nms[idx];

        auto unmap_x = [&](float cx_) -> int {
            return std::max(0, std::min(
                static_cast<int>((cx_ - pad_x) / scale), image.cols - 1));
        };
        auto unmap_y = [&](float cy_) -> int {
            return std::max(0, std::min(
                static_cast<int>((cy_ - pad_y) / scale), image.rows - 1));
        };

        int x1 = unmap_x(static_cast<float>(b.x));
        int y1 = unmap_y(static_cast<float>(b.y));
        int x2 = unmap_x(static_cast<float>(b.x + b.width));
        int y2 = unmap_y(static_cast<float>(b.y + b.height));

        if (x2 <= x1 || y2 <= y1) continue;

        detections.push_back({
            cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)),
            scores_nms[idx],
            class_nms[idx]
        });
    }
    return detections;
}
