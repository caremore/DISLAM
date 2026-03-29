#ifndef SUPERPOINTEXTRACTOR_H
#define SUPERPOINTEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <mutex>

namespace ORB_SLAM3
{

class SuperPointExtractor
{
public:
    SuperPointExtractor(int nfeatures, float threshold, const std::string& sp_onnx_path,
                       const std::string& netvlad_onnx_path = "", bool use_cuda = false);

    SuperPointExtractor(int nfeatures, float threshold, float scaleFactor, 
                       int nlevels, const std::string& sp_onnx_path,
                       const std::string& netvlad_onnx_path = "", bool use_cuda = false);

    ~SuperPointExtractor(){}

    // Compute the features and descriptors on an image - 与 HFextractor 完全相同的接口
    int operator()(const cv::Mat &_image, std::vector<cv::KeyPoint>& _keypoints,
                   cv::Mat &_localDescriptors);
    // 新增：获取全局描述子的接口
    void getGlobalDescriptor(const cv::Mat &_image,cv::Mat &_globalDescriptors);

    int inline GetLevels(void) {
        return nlevels;}

    float inline GetScaleFactor(void) {
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(void) {
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(void) {
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(void) {
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(void) {
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;
    std::vector<int> mnFeaturesPerLevel;

    int nfeatures;
    float threshold;

    // 模型状态检查
    bool IsModelLoaded() const;
    bool IsNetVLADLoaded() const;

protected:
    double scaleFactor;
    int nlevels;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    // ONNX Runtime 相关
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::MemoryInfo memory_info_;
    std::unique_ptr<Ort::Session> sp_session_;
    std::unique_ptr<Ort::Session> netvlad_session_;
    
    // 线程安全
    mutable std::mutex inference_mutex_;

    // 模型路径
    std::string sp_onnx_path_;
    std::string netvlad_onnx_path_;

    void ComputePyramid(const cv::Mat &image);

    int ExtractSingleLayer(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                           cv::Mat &localDescriptors);

    int ExtractMultiLayers(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                           cv::Mat &localDescriptors);

    // SuperPoint 推理方法
    void RunSuperPoint(const cv::Mat& gray,
                      int maxKpts, float scoreThresh,
                      std::vector<cv::KeyPoint>& keypoints,
                      cv::Mat& descriptors);

    void RunSuperPointWithGlobal(const cv::Mat& gray,
                                int maxKpts, float scoreThresh,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::Mat& descriptors,
                                cv::Mat& globalDescriptors);

    cv::Mat RunNetVLAD(const cv::Mat& bgr);

    // 工具方法
    std::vector<float> MatToCHW(const cv::Mat& gray);
    cv::Mat DecodeHeatmapFromSemi(float* semi_data, int Hc, int Wc, int H, int W);
    void NMS(const cv::Mat& score, int r, float thresh,
            std::vector<cv::Point2f>& pts, std::vector<float>& conf, int topk = -1);
    cv::Mat SampleDesc(float* desc_data, int Hc, int Wc, int D,
                      const std::vector<cv::Point2f>& pts, int H, int W);
};

} //namespace ORB_SLAM3

#endif // SUPERPOINTEXTRACTOR_H