#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include "faster_rcnn_ros/faster_rcnn_detector.hpp"

class FasterRCNNNode : public rclcpp::Node {
public:
    FasterRCNNNode() : Node("faster_rcnn_node") {
        declare_parameter<std::string>("engine_path", "");
        declare_parameter<std::string>("labels_path", "");
        declare_parameter<std::string>("input_topic", "/camera/image_raw");
        declare_parameter<std::string>("overlay_topic", "/detectnet/overlay");
        declare_parameter<double>("threshold", 0.5);
        declare_parameter<int>("input_height", 800);
        declare_parameter<int>("input_width", 1344);

        auto engine_path = get_parameter("engine_path").as_string();
        auto labels_path = get_parameter("labels_path").as_string();
        auto input_topic = get_parameter("input_topic").as_string();
        auto overlay_topic = get_parameter("overlay_topic").as_string();
        threshold_ = get_parameter("threshold").as_double();
        int input_h = get_parameter("input_height").as_int();
        int input_w = get_parameter("input_width").as_int();

        detector_ = std::make_unique<FasterRCNNDetector>(engine_path, labels_path, input_h, input_w);

        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_topic, 10,
            std::bind(&FasterRCNNNode::imageCallback, this, std::placeholders::_1)
        );

        overlay_pub_ = this->create_publisher<sensor_msgs::msg::Image>(overlay_topic, 10);

        RCLCPP_INFO(this->get_logger(), "Faster R-CNN node started.");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            auto detections = detector_->infer(cv_ptr->image, static_cast<float>(threshold_));

            cv::Mat overlay = cv_ptr->image.clone();
            for (const auto& det : detections) {
                cv::rectangle(overlay, det.box, cv::Scalar(0, 255, 0), 2);
                std::string label = (det.class_id < detector_->getClassNames().size())
                                    ? detector_->getClassNames()[det.class_id]
                                    : std::to_string(det.class_id);
                std::string text = label + ": " + std::to_string(det.confidence).substr(0, 4);
                cv::putText(overlay, text, cv::Point(det.box.x, det.box.y - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }

            auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", overlay).toImageMsg();
            overlay_pub_->publish(*out_msg);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error in callback: %s", e.what());
        }
    }

    std::unique_ptr<FasterRCNNDetector> detector_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;
    double threshold_ = 0.5;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FasterRCNNNode>());
    rclcpp::shutdown();
    return 0;
}