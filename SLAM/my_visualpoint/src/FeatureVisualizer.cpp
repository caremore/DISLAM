#include "FeatureVisualizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>


FeatureVisualizer::FeatureVisualizer() {
}

bool FeatureVisualizer::LoadFeatureData(const std::string& data_dir) {
    namespace fs = std::filesystem;
    
    if (!fs::exists(data_dir)) {
        std::cerr << "Data directory does not exist: " << data_dir << std::endl;
        return false;
    }
    
    frames_data_.clear();
    
    // 查找所有的特征点文件
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".txt") {
            std::string filename = entry.path().stem().string();
            
            // 解析帧ID (假设文件名格式: frame_XXXX_keypoints.txt)
            if (filename.find("frame_") != std::string::npos && 
                filename.find("_keypoints") != std::string::npos) {
                
                std::string frame_str = filename.substr(6, filename.find("_keypoints") - 6);
                try {
                    int frame_id = std::stoi(frame_str);
                    
                    FrameData frame_data;
                    frame_data.frame_id = frame_id;
                    frame_data.image_path = data_dir + "/frame_" + std::to_string(frame_id) + ".png";
                    
                    if (ParseKeypointsFile(entry.path().string(), frame_data)) {
                        frames_data_[frame_id] = frame_data;
                        std::cout << "Loaded frame " << frame_id << " with " 
                                  << frame_data.keypoints.size() << " features" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing frame ID from: " << filename << std::endl;
                }
            }
        }
    }
    
    std::cout << "Total frames loaded: " << frames_data_.size() << std::endl;
    return !frames_data_.empty();
}

bool FeatureVisualizer::ParseKeypointsFile(const std::string& filename, FrameData& frame_data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open keypoints file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    std::getline(file, line); // 跳过标题行
    
    int count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        FeaturePoint kp;
        
        try {
            // 解析 CSV 格式: x,y,size,angle,response,octave,class_id
            std::getline(ss, token, ',');
            kp.x = std::stof(token);
            std::getline(ss, token, ',');
            kp.y = std::stof(token);
            std::getline(ss, token, ',');
            kp.size = std::stof(token);
            std::getline(ss, token, ',');
            kp.angle = std::stof(token);
            std::getline(ss, token, ',');
            kp.response = std::stof(token);
            std::getline(ss, token, ',');
            kp.octave = std::stoi(token);
            std::getline(ss, token, ',');
            kp.class_id = std::stoi(token);
            
            frame_data.keypoints.push_back(kp);
            count++;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing keypoint data: " << line << std::endl;
        }
    }
    
    file.close();
    std::cout << "Parsed " << count << " keypoints from " << filename << std::endl;
    return count > 0;
}

bool FeatureVisualizer::LoadImage(FrameData& frame_data) {
    if (!frame_data.image.empty()) {
        return true;
    }
    
    frame_data.image = cv::imread(frame_data.image_path, cv::IMREAD_COLOR);
    if (frame_data.image.empty()) {
        std::cerr << "Cannot load image: " << frame_data.image_path << std::endl;
        return false;
    }
    
    return true;
}

cv::Mat FeatureVisualizer::CreateORBSLAM3Visualization(const FrameData& frame_data) {
    if (!LoadImage(frame_data)) {
        return cv::Mat();
    }
    
    // 创建副本，不修改原图
    cv::Mat visualization = frame_data.image.clone();
    
    // ORB-SLAM3风格：绿色特征点，简洁绘制
    for (const auto& kp : frame_data.keypoints) {
        cv::Point2f center(kp.x, kp.y);
        
        // 绘制特征点圆圈 - 绿色，线宽1
        cv::circle(visualization, center, 3, cv::Scalar(0, 255, 0), 1);
        
        // 如果特征点有方向，绘制方向线
        if (kp.angle >= 0) {
            float angle_rad = kp.angle * CV_PI / 180.0f;
            float line_length = 6.0f; // 方向线长度
            cv::Point2f direction(cos(angle_rad) * line_length, sin(angle_rad) * line_length);
            cv::line(visualization, center, center + direction, cv::Scalar(0, 255, 0), 1);
        }
    }
    
    // 添加简单的信息文本（ORB-SLAM3风格）
    std::string info = "Frame: " + std::to_string(frame_data.frame_id) + 
                      "  Features: " + std::to_string(frame_data.keypoints.size());
    
    cv::putText(visualization, info, cv::Point(10, 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    cv::putText(visualization, info, cv::Point(10, 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    
    return visualization;
}

void FeatureVisualizer::VisualizeFrame(int frame_id, bool show_window) {
    auto it = frames_data_.find(frame_id);
    if (it == frames_data_.end()) {
        std::cerr << "Frame " << frame_id << " not found!" << std::endl;
        return;
    }
    
    cv::Mat visualization = CreateORBSLAM3Visualization(it->second);
    if (visualization.empty()) {
        std::cerr << "Failed to create visualization for frame " << frame_id << std::endl;
        return;
    }
    
    if (show_window) {
        std::string window_name = "Frame " + std::to_string(frame_id);
        cv::imshow(window_name, visualization);
        std::cout << "Displaying frame " << frame_id << ". Press any key to continue..." << std::endl;
        cv::waitKey(0);
        cv::destroyWindow(window_name);
    }
}

void FeatureVisualizer::SaveAllFrames(const std::string& output_dir) {
    // 创建输出目录
    struct stat st;
    if (stat(output_dir.c_str(), &st) != 0) {
        // 目录不存在，创建它
        mkdir(output_dir.c_str(), 0755);
    }
    
    int saved_count = 0;
    for (auto it = frames_data_.begin(); it != frames_data_.end(); ++it) {
        int frame_id = it->first;
        FrameData& frame_data = it->second;  // 非 const 引用
        
        // 确保图像已加载
        if (!LoadImage(frame_data)) {
            std::cerr << "Failed to load image for frame " << frame_id << std::endl;
            continue;
        }
        
        cv::Mat visualization = CreateORBSLAM3Visualization(frame_data);
        if (!visualization.empty()) {
            std::string output_path = output_dir + "/frame_" + std::to_string(frame_id) + "_features.png";
            if (cv::imwrite(output_path, visualization)) {
                std::cout << "Saved: " << output_path << std::endl;
                saved_count++;
            } else {
                std::cerr << "Failed to save: " << output_path << std::endl;
            }
        }
    }
    
    std::cout << "Successfully saved " << saved_count << " frames to " << output_dir << std::endl;
}

std::vector<int> FeatureVisualizer::GetFrameList() const {
    std::vector<int> frame_list;
    for (const auto& [frame_id, _] : frames_data_) {
        frame_list.push_back(frame_id);
    }
    std::sort(frame_list.begin(), frame_list.end());
    return frame_list;
}
