#ifndef FEATURE_VISUALIZER_H
#define FEATURE_VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>

struct FeaturePoint {
    float x, y;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;
};

struct FrameData {
    int frame_id;
    std::string image_path;
    std::vector<FeaturePoint> keypoints;
    cv::Mat image;
};

class FeatureVisualizer {
public:
    FeatureVisualizer();
    
    // 加载特征数据
    bool LoadFeatureData(const std::string& data_dir);
    
    // 可视化特定帧（ORB-SLAM3风格）
    void VisualizeFrame(int frame_id, bool show_window = true);
    
    // 批量保存可视化结果
    void SaveAllFrames(const std::string& output_dir);
    
    // 获取帧列表
    std::vector<int> GetFrameList() const;

private:
    std::map<int, FrameData> frames_data_;
    
    // 辅助函数
    bool LoadImage(FrameData& frame_data);
    bool ParseKeypointsFile(const std::string& filename, FrameData& frame_data);
    cv::Mat CreateORBSLAM3Visualization(const FrameData& frame_data) const;
};

#endif // FEATURE_VISUALIZER_H
