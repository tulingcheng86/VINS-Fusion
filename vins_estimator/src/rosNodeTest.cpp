/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <visualization_msgs/MarkerArray.h>
#include "estimator/estimator.h"
#include "estimator/parameters.h"
#include "utility/visualization.h"
#include <yolo_ros/DetectionMessages.h>

Estimator estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

// 全局变量，用于存储最新的YOLO检测消息
// yolo_ros::DetectionMessages::ConstPtr latest_detection_msgs;
// std::mutex m_yolo;

yolo_ros::DetectionMessages::ConstPtr latest_detection_msgs_left;
yolo_ros::DetectionMessages::ConstPtr latest_detection_msgs_right;
std::mutex m_yolo_left;
std::mutex m_yolo_right;

ros::Publisher marker_pub;

// 结构体来存储三维点和标签
struct ObjectDetection {
    Eigen::Vector3d point; // 使用三维点
    std::string label;
};


// 全局数据结构，用于存储物体信息
std::map<std::string, std::vector<Eigen::Vector4d>> detected_objects;
std::mutex detected_objects_mutex;

// 添加物体到三维地图
void add_to_3d_map(const Eigen::Vector4d& point, const std::string& label) {
    detected_objects[label].push_back(point);
}

//single_camera
// void yolo_callback(const yolo_ros::DetectionMessages::ConstPtr& msg) {
//     std::lock_guard<std::mutex> lock(m_yolo);
//     latest_detection_msgs = msg;
// }

void yolo_callback_left(const yolo_ros::DetectionMessages::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(m_yolo_left);
    latest_detection_msgs_left = msg;
}

void yolo_callback_right(const yolo_ros::DetectionMessages::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(m_yolo_right);
    latest_detection_msgs_right = msg;
}

// 在drawDetections函数之前声明visualize_detections函数
void visualize_detections(cv::Mat& image, const Eigen::Matrix4d& transform);

void drawDetections(cv::Mat& img, const yolo_ros::DetectionMessages::ConstPtr& detection_msgs, const Eigen::Matrix4d& transform) {
    if (detection_msgs) {
        for (const auto& dmsg : detection_msgs->data) {
            cv::Point p1(dmsg.x1, dmsg.y1), p2(dmsg.x2, dmsg.y2);
            cv::rectangle(img, p1, p2, cv::Scalar(255, 0, 0), 2);
            cv::putText(img, dmsg.label, p1, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1);
            
            Eigen::Vector4d point_2d(dmsg.x1, dmsg.y1, 0, 1);  // 假设深度为0
            Eigen::Vector4d point_3d = transform * point_2d;
            add_to_3d_map(point_3d, dmsg.label);
        }
    }
    
    visualize_detections(img, transform);
}

void visualize_detections(cv::Mat& image, const Eigen::Matrix4d& transform) {
    std::lock_guard<std::mutex> lock(detected_objects_mutex);

    visualization_msgs::MarkerArray marker_array;
    int id = 0; // Marker ID

    for (const auto& obj : detected_objects) {
        for (const auto& point : obj.second) {
            Eigen::Vector4d transformed_point = transform.inverse() * point;
            visualization_msgs::Marker marker;

            marker.header.frame_id = "world"; // Replace with your fixed frame
            marker.header.stamp = ros::Time::now();
            marker.ns = "detections";
            marker.id = id++;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            marker.pose.position.x = transformed_point.x();
            marker.pose.position.y = transformed_point.y();
            marker.pose.position.z = transformed_point.z();
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = 0.1;
            marker.scale.y = 0.1;
            marker.scale.z = 0.1;

            marker.color.a = 1.0;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;

            marker_array.markers.push_back(marker);

            cv::Point2d img_point(transformed_point.x() / transformed_point.z(), transformed_point.y() / transformed_point.z());
            cv::putText(image, obj.first, img_point, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 255), 1);
        }
    }

    marker_pub.publish(marker_array);
}


void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}


cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

// 修改sync_process函数，传递变换矩阵到drawDetections
void sync_process()
{
    while (1)
    {
        if (STEREO)
        {
            cv::Mat image0, image1;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty() && !img1_buf.empty())
            {
                double time0 = img0_buf.front()->header.stamp.toSec();
                double time1 = img1_buf.front()->header.stamp.toSec();
                if (time0 < time1 - 0.003)
                {
                    img0_buf.pop();
                    printf("throw img0\n");
                }
                else if (time0 > time1 + 0.003)
                {
                    img1_buf.pop();
                    printf("throw img1\n");
                }
                else
                {
                    time = img0_buf.front()->header.stamp.toSec();
                    header = img0_buf.front()->header;
                    image0 = getImageFromMsg(img0_buf.front());
                    img0_buf.pop();
                    image1 = getImageFromMsg(img1_buf.front());
                    img1_buf.pop();
                }
            }
            m_buf.unlock();
            if (!image0.empty()) {
                Eigen::Matrix4d transform;
                estimator.getPoseInWorldFrame(transform);
                {
                    std::lock_guard<std::mutex> lock(m_yolo_left);
                    drawDetections(image0, latest_detection_msgs_left, transform);
                }
                {
                    std::lock_guard<std::mutex> lock(m_yolo_right);
                    drawDetections(image1, latest_detection_msgs_right, transform);
                }
                visualize_detections(image0, transform);
                visualize_detections(image1, transform);
                estimator.inputImage(time, image0, image1);
            }
        }
        else
        {
            cv::Mat image;
            std_msgs::Header header;
            double time = 0;
            m_buf.lock();
            if (!img0_buf.empty())
            {
                time = img0_buf.front()->header.stamp.toSec();
                header = img0_buf.front()->header;
                image = getImageFromMsg(img0_buf.front());
                img0_buf.pop();
            }
            m_buf.unlock();
            if (!image.empty()) {
                Eigen::Matrix4d transform;
                estimator.getPoseInWorldFrame(transform);
                {
                    std::lock_guard<std::mutex> lock(m_yolo_left);
                    drawDetections(image, latest_detection_msgs_left, transform);
                }
                {
                    std::lock_guard<std::mutex> lock(m_yolo_right);
                    drawDetections(image, latest_detection_msgs_right, transform);
                }
                visualize_detections(image, transform);
                estimator.inputImage(time, image);
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}


void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator.inputIMU(t, acc, gyr);
    return;
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (unsigned int i = 0; i < feature_msg->points.size(); i++)
    {
        int feature_id = feature_msg->channels[0].values[i];
        int camera_id = feature_msg->channels[1].values[i];
        double x = feature_msg->points[i].x;
        double y = feature_msg->points[i].y;
        double z = feature_msg->points[i].z;
        double p_u = feature_msg->channels[2].values[i];
        double p_v = feature_msg->channels[3].values[i];
        double velocity_x = feature_msg->channels[4].values[i];
        double velocity_y = feature_msg->channels[5].values[i];
        if(feature_msg->channels.size() > 5)
        {
            double gx = feature_msg->channels[6].values[i];
            double gy = feature_msg->channels[7].values[i];
            double gz = feature_msg->channels[8].values[i];
            pts_gt[feature_id] = Eigen::Vector3d(gx, gy, gz);
            //printf("receive pts gt %d %f %f %f\n", feature_id, gx, gy, gz);
        }
        ROS_ASSERT(z == 1);
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }
    double t = feature_msg->header.stamp.toSec();
    estimator.inputFeature(t, featureFrame);
    return;
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        estimator.clearState();
        estimator.setParameter();
    }
    return;
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use IMU!");
        estimator.changeSensorType(1, STEREO);
    }
    else
    {
        //ROS_WARN("disable IMU!");
        estimator.changeSensorType(0, STEREO);
    }
    return;
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        //ROS_WARN("use stereo!");
        estimator.changeSensorType(USE_IMU, 1);
    }
    else
    {
        //ROS_WARN("use mono camera (left)!");
        estimator.changeSensorType(USE_IMU, 0);
    }
    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2)
    {
        printf("please intput: rosrun vins vins_node [config file] \n"
               "for example: rosrun vins vins_node "
               "~/catkin_ws/src/VINS-Fusion/config/euroc/euroc_stereo_imu_config.yaml \n");
        return 1;
    }

    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    readParameters(config_file);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    registerPub(n);

    ros::Subscriber sub_imu;
    if(USE_IMU)
    {
        sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    }
    ros::Subscriber sub_feature = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    ros::Subscriber sub_img0 = n.subscribe(IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1;
    if(STEREO)
    {
        sub_img1 = n.subscribe(IMAGE1_TOPIC, 100, img1_callback);
    }
    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

    // Initialize the marker publisher
    marker_pub = n.advertise<visualization_msgs::MarkerArray>("detection_markers", 1);

    // 订阅YOLO检测结果
    ros::Subscriber sub_yolo_left = n.subscribe("/untracked_info_left", 10, yolo_callback_left);
    ros::Subscriber sub_yolo_right = n.subscribe("/untracked_info_right", 10, yolo_callback_right);

    std::thread sync_thread{sync_process};
    ros::spin();

    return 0;
}
