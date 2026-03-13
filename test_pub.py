#!/usr/bin/env python3
"""
发一张测试图到 /detectnet/image_in，监听 /detectnet/overlay 的回包，
打印推理延迟和结果（仅验证用，用后可删）。
不依赖 cv_bridge，直接手动构造 sensor_msgs/Image。
"""
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time

IMAGE_PATH = "/home/nvidia/ros2_ws/test_images/nuscenes_sample.jpg"
PUB_TOPIC  = "/detectnet/image_in"
SUB_TOPIC  = "/detectnet/overlay"


def mat_to_imgmsg(img_bgr: np.ndarray) -> Image:
    """BGR OpenCV Mat → sensor_msgs/Image (bgr8)，无需 cv_bridge"""
    msg = Image()
    msg.height = img_bgr.shape[0]
    msg.width  = img_bgr.shape[1]
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = img_bgr.shape[1] * 3
    msg.data = img_bgr.tobytes()
    return msg


def imgmsg_to_mat(msg: Image) -> np.ndarray:
    """sensor_msgs/Image (bgr8) → BGR OpenCV Mat，无需 cv_bridge"""
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return arr.reshape((msg.height, msg.width, 3))


class TestPub(Node):
    def __init__(self):
        super().__init__("test_pub")
        self.pub = self.create_publisher(Image, PUB_TOPIC, 10)
        self.sub = self.create_subscription(Image, SUB_TOPIC, self.cb, 10)
        self.received = False
        self.t_send = None
        # 等 faster_rcnn_node 订阅者就绪后发图
        self.timer = self.create_timer(1.5, self.send_once)

    def send_once(self):
        self.timer.cancel()
        img = cv2.imread(IMAGE_PATH)
        if img is None:
            self.get_logger().error(f"无法读取图片: {IMAGE_PATH}")
            rclpy.shutdown(); return
        msg = mat_to_imgmsg(img)
        self.t_send = time.time()
        self.pub.publish(msg)
        self.get_logger().info(
            f"已发送 {img.shape[1]}x{img.shape[0]} 图片，等待推理结果..."
        )
        self.timeout_timer = self.create_timer(15.0, self.on_timeout)

    def cb(self, msg: Image):
        if self.received:
            return
        self.received = True
        latency = (time.time() - self.t_send) * 1000
        self.get_logger().info(
            f"✅ 收到 overlay 帧!  推理延迟: {latency:.0f} ms  "
            f"({1000/latency:.1f} fps)"
        )
        out = imgmsg_to_mat(msg)
        cv2.imwrite("/tmp/overlay_result.jpg", out)
        self.get_logger().info("检测结果已保存到 /tmp/overlay_result.jpg")
        rclpy.shutdown()

    def on_timeout(self):
        if not self.received:
            self.get_logger().error("⚠️  15 秒内未收到 overlay，可能未触发推理")
        rclpy.shutdown()


def main():
    rclpy.init()
    rclpy.spin(TestPub())


if __name__ == "__main__":
    main()
