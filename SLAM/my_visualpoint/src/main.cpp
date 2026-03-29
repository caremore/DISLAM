#include "FeatureVisualizer.h"
#include <iostream>
#include <string>
#include <algorithm>

void PrintUsage() {
    std::cout << "ORB-SLAM3 Style Feature Visualizer" << std::endl;
    std::cout << "Usage: feature_visualizer [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -d, --data DIR    : Data directory containing feature files" << std::endl;
    std::cout << "  -f, --frame ID    : Visualize specific frame" << std::endl;
    std::cout << "  -a, --all         : Save visualizations for all frames" << std::endl;
    std::cout << "  -o, --output DIR  : Output directory for saved images" << std::endl;
    std::cout << "  -l, --list        : List all available frames" << std::endl;
    std::cout << "  -h, --help        : Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    std::string data_dir = "data";
    std::vector<int> frame_ids;
    bool save_all = false;
    bool list_frames = false;
    std::string output_dir = "output";
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-d" || arg == "--data") {
            if (i + 1 < argc) {
                data_dir = argv[++i];
            } else {
                std::cerr << "Error: --data requires directory path" << std::endl;
                return 1;
            }
        } else if (arg == "-f" || arg == "--frame") {
            if (i + 1 < argc) {
                try {
                    frame_ids.push_back(std::stoi(argv[++i]));
                } catch (const std::exception& e) {
                    std::cerr << "Error: invalid frame ID" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: --frame requires frame ID" << std::endl;
                return 1;
            }
        } else if (arg == "-a" || arg == "--all") {
            save_all = true;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_dir = argv[++i];
            } else {
                std::cerr << "Error: --output requires directory path" << std::endl;
                return 1;
            }
        } else if (arg == "-l" || arg == "--list") {
            list_frames = true;
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage();
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            PrintUsage();
            return 1;
        }
    }
    
    // 创建可视化器
    FeatureVisualizer visualizer;
    
    // 加载数据
    if (!visualizer.LoadFeatureData(data_dir)) {
        std::cerr << "Failed to load feature data from: " << data_dir << std::endl;
        return 1;
    }
    
    // 列出所有帧
    if (list_frames) {
        auto frame_list = visualizer.GetFrameList();
        std::cout << "Available frames (" << frame_list.size() << "): ";
        for (size_t i = 0; i < frame_list.size(); ++i) {
            std::cout << frame_list[i];
            if (i < frame_list.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        return 0;
    }
    
    // 执行操作
    if (save_all) {
        std::cout << "Saving visualizations for all frames..." << std::endl;
        visualizer.SaveAllFrames(output_dir);
    } else if (!frame_ids.empty()) {
        for (int frame_id : frame_ids) {
            std::cout << "Visualizing frame " << frame_id << "..." << std::endl;
            visualizer.VisualizeFrame(frame_id, true);
        }
    } else {
        std::cout << "No operation specified. Use -h for help." << std::endl;
        PrintUsage();
    }
    
    return 0;
}
