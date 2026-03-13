#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include "faster_rcnn_ros/faster_rcnn_detector.hpp"

namespace fs = std::filesystem;

// ─────────────────── 辅助：在帧上绘制检测结果 ───────────────────────────────
static void drawDetections(cv::Mat& img,
                           const std::vector<Detection>& dets,
                           const std::vector<std::string>& class_names)
{
    for (const auto& det : dets) {
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = (det.class_id < static_cast<int>(class_names.size()))
                            ? class_names[det.class_id]
                            : std::to_string(det.class_id);
        std::string text = label + ": " + std::to_string(det.confidence).substr(0, 4);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        int ty = std::max(det.box.y - 5, ts.height + 2);
        cv::putText(img, text, cv::Point(det.box.x, ty),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

// ─────────────────────────── 主节点 ────────────────────────────────────────
class FasterRCNNNode : public rclcpp::Node {
public:
    FasterRCNNNode() : Node("faster_rcnn_node") {
        // ── 参数声明 ────────────────────────────────────────────────────────
        declare_parameter<std::string>("engine_path", "");
        declare_parameter<std::string>("labels_path", "");
        declare_parameter<std::string>("input_topic",  "/camera/image_raw");
        declare_parameter<std::string>("overlay_topic", "/detectnet/overlay");
        declare_parameter<double>("threshold", 0.5);
        declare_parameter<int>("input_height", 375);
        declare_parameter<int>("input_width",  1242);
        // 文件模式专用参数
        declare_parameter<std::string>("input_path",  "");   // 图片文件 / 视频文件 / 图片目录
        declare_parameter<std::string>("output_path", "");   // 结果输出路径（文件或目录）
        declare_parameter<bool>("loop_video", false);         // 视频是否循环

        // ── 读取参数 ────────────────────────────────────────────────────────
        auto engine_path   = get_parameter("engine_path").as_string();
        auto labels_path   = get_parameter("labels_path").as_string();
        auto input_topic   = get_parameter("input_topic").as_string();
        auto overlay_topic = get_parameter("overlay_topic").as_string();
        threshold_    = get_parameter("threshold").as_double();
        int input_h   = get_parameter("input_height").as_int();
        int input_w   = get_parameter("input_width").as_int();
        input_path_   = get_parameter("input_path").as_string();
        output_path_  = get_parameter("output_path").as_string();
        loop_video_   = get_parameter("loop_video").as_bool();

        // ── 初始化检测器 ────────────────────────────────────────────────────
        detector_ = std::make_unique<FasterRCNNDetector>(engine_path, labels_path, input_h, input_w);

        // ── 结果话题（两种模式都发布）──────────────────────────────────────
        overlay_pub_ = create_publisher<sensor_msgs::msg::Image>(overlay_topic, 10);

        if (input_path_.empty()) {
            // ══ 话题订阅模式（原有逻辑，不变）══════════════════════════════
            subscription_ = create_subscription<sensor_msgs::msg::Image>(
                input_topic, 10,
                std::bind(&FasterRCNNNode::topicCallback, this, std::placeholders::_1));
            RCLCPP_INFO(get_logger(), "[话题模式] 订阅: %s → 发布: %s",
                        input_topic.c_str(), overlay_topic.c_str());
        } else {
            // ══ 文件模式 ════════════════════════════════════════════════════
            initFileMode();
        }
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    // 话题模式：接收 sensor_msgs/Image → 推理 → 发布结果话题
    // ─────────────────────────────────────────────────────────────────────────
    void topicCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            auto detections = detector_->infer(cv_ptr->image, static_cast<float>(threshold_));

            cv::Mat overlay = cv_ptr->image.clone();
            drawDetections(overlay, detections, detector_->getClassNames());

            overlay_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "[话题模式] 错误: %s", e.what());
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 文件模式初始化：判断路径类型，准备文件列表 / VideoCapture / VideoWriter
    // ─────────────────────────────────────────────────────────────────────────
    void initFileMode() {
        fs::path p(input_path_);

        if (fs::is_directory(p)) {
            // ── 目录：收集所有图片 ──────────────────────────────────────────
            const std::vector<std::string> IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff"};
            for (auto& e : fs::directory_iterator(p)) {
                std::string ext = e.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (std::find(IMG_EXTS.begin(), IMG_EXTS.end(), ext) != IMG_EXTS.end())
                    image_files_.push_back(e.path().string());
            }
            std::sort(image_files_.begin(), image_files_.end());
            if (!output_path_.empty()) fs::create_directories(output_path_);
            RCLCPP_INFO(get_logger(), "[目录模式] 共 %zu 张图片，输出目录: %s",
                        image_files_.size(), output_path_.empty() ? "(不保存)" : output_path_.c_str());
            // 立即开始，0ms 间隔（处理完一张立刻处理下一张）
            file_timer_ = create_wall_timer(std::chrono::milliseconds(0),
                              std::bind(&FasterRCNNNode::fileTimerCallback, this));

        } else {
            // ── 判断是视频还是单张图片 ──────────────────────────────────────
            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            const std::vector<std::string> VID_EXTS = {".mp4",".avi",".mov",".mkv",".m4v",".flv"};
            bool is_video = std::find(VID_EXTS.begin(), VID_EXTS.end(), ext) != VID_EXTS.end();

            if (is_video) {
                // ── 视频模式 ────────────────────────────────────────────────
                cap_ = std::make_unique<cv::VideoCapture>(input_path_);
                if (!cap_->isOpened())
                    throw std::runtime_error("无法打开视频: " + input_path_);

                fps_ = cap_->get(cv::CAP_PROP_FPS);
                if (fps_ <= 0.0) fps_ = 25.0;
                total_frames_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_COUNT));
                int vid_w = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
                int vid_h = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));

                if (!output_path_.empty()) {
                    // 使用 mp4v 编码写出结果视频
                    int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
                    writer_ = std::make_unique<cv::VideoWriter>(
                        output_path_, fourcc, fps_, cv::Size(vid_w, vid_h));
                    if (!writer_->isOpened())
                        RCLCPP_WARN(get_logger(), "无法打开输出视频: %s", output_path_.c_str());
                }
                RCLCPP_INFO(get_logger(),
                    "[视频模式] %s | %.1f fps | %d 帧 | %dx%d | 输出: %s",
                    input_path_.c_str(), fps_, total_frames_, vid_w, vid_h,
                    output_path_.empty() ? "(不保存)" : output_path_.c_str());

                int interval_ms = std::max(1, static_cast<int>(1000.0 / fps_));
                file_timer_ = create_wall_timer(
                    std::chrono::milliseconds(interval_ms),
                    std::bind(&FasterRCNNNode::fileTimerCallback, this));

            } else {
                // ── 单张图片模式 ─────────────────────────────────────────────
                image_files_.push_back(input_path_);
                RCLCPP_INFO(get_logger(), "[图片模式] %s → 输出: %s",
                    input_path_.c_str(),
                    output_path_.empty() ? "(不保存)" : output_path_.c_str());
                file_timer_ = create_wall_timer(std::chrono::milliseconds(0),
                                  std::bind(&FasterRCNNNode::fileTimerCallback, this));
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 文件模式 Timer 回调：每次处理一帧（图片或视频帧）
    // ─────────────────────────────────────────────────────────────────────────
    void fileTimerCallback() {
        cv::Mat frame;
        std::string frame_name;

        if (cap_) {
            // 视频：读下一帧
            *cap_ >> frame;
            if (frame.empty()) {
                if (loop_video_) {
                    cap_->set(cv::CAP_PROP_POS_FRAMES, 0);
                    *cap_ >> frame;
                }
                if (frame.empty()) {
                    RCLCPP_INFO(get_logger(), "[视频模式] 处理完成，共 %d 帧", frame_index_);
                    if (writer_) writer_->release();
                    file_timer_->cancel();
                    rclcpp::shutdown();
                    return;
                }
            }
            frame_index_++;
            frame_name = "frame_" + std::to_string(frame_index_);
            if (total_frames_ > 0 && frame_index_ % 100 == 0) {
                RCLCPP_INFO(get_logger(), "[视频模式] 进度: %d / %d (%.1f%%)",
                    frame_index_, total_frames_,
                    100.0 * frame_index_ / total_frames_);
            }
        } else {
            // 图片/目录：读当前索引
            if (file_index_ >= image_files_.size()) {
                RCLCPP_INFO(get_logger(), "[图片模式] 所有图片处理完成，共 %zu 张",
                            image_files_.size());
                file_timer_->cancel();
                rclcpp::shutdown();
                return;
            }
            const std::string& img_path = image_files_[file_index_++];
            frame = cv::imread(img_path);
            if (frame.empty()) {
                RCLCPP_WARN(get_logger(), "无法读取图片，跳过: %s", img_path.c_str());
                return;
            }
            frame_name = fs::path(img_path).filename().string();
        }

        // ── 推理 ──────────────────────────────────────────────────────────
        auto t0 = std::chrono::steady_clock::now();
        auto detections = detector_->infer(frame, static_cast<float>(threshold_));
        double elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        // ── 绘制结果 ──────────────────────────────────────────────────────
        cv::Mat overlay = frame.clone();
        drawDetections(overlay, detections, detector_->getClassNames());

        RCLCPP_INFO(get_logger(), "[%s] %zu 个目标 | %.1f ms",
                    frame_name.c_str(), detections.size(), elapsed_ms);

        // ── 保存结果 ──────────────────────────────────────────────────────
        if (!output_path_.empty()) {
            if (writer_ && writer_->isOpened()) {
                // 视频
                writer_->write(overlay);
            } else {
                // 图片：output_path 是文件（单张）或目录（批量）
                std::string out_file;
                if (image_files_.size() == 1) {
                    out_file = output_path_;   // 单张：直接写到指定路径
                } else {
                    out_file = (fs::path(output_path_) /
                                ("result_" + fs::path(frame_name).filename().string())).string();
                }
                cv::imwrite(out_file, overlay);
            }
        }

        // ── 发布到 ROS2 话题（可选，供 rviz / bag 录制使用）──────────────
        std_msgs::msg::Header hdr;
        hdr.stamp = this->now();
        auto ros_msg = cv_bridge::CvImage(hdr, "bgr8", overlay).toImageMsg();
        overlay_pub_->publish(*ros_msg);
    }

    // ─── 成员变量 ──────────────────────────────────────────────────────────
    std::unique_ptr<FasterRCNNDetector>  detector_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;  // 话题模式
    rclcpp::TimerBase::SharedPtr file_timer_;                                // 文件模式

    // 文件模式状态
    std::string input_path_;
    std::string output_path_;
    bool        loop_video_   = false;
    std::vector<std::string> image_files_;
    size_t      file_index_   = 0;
    int         frame_index_  = 0;
    int         total_frames_ = 0;
    double      fps_          = 25.0;
    std::unique_ptr<cv::VideoCapture> cap_;
    std::unique_ptr<cv::VideoWriter>  writer_;

    double threshold_ = 0.5;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FasterRCNNNode>());
    rclcpp::shutdown();
    return 0;
}