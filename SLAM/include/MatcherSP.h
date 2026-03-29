#ifndef MATCHER_H
#define MATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>
#include"sophus/sim3.hpp"

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"


namespace ORB_SLAM3 {

class MatcherSP {

    public:
        // 构造函数
        MatcherSP();

        // ==================== 静态工具方法 ====================
        
        /**
         * @brief 计算两个描述子之间的距离
         * @param a 描述子a
         * @param b 描述子b  
         * @return 距离值
         * 
         * 使用Eigen加速的L2距离计算，比OpenCV原生的norm函数更快
         */
        static float DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

        // ==================== 投影匹配方法族 ====================

        /**
         * @brief 在帧中搜索与投影地图点的匹配
         * @param F 当前帧
         * @param vpMapPoints 地图点列表
         * @param fNNratio 最近邻比率阈值
         * @param th 搜索半径系数
         * @param bFarPoints 是否处理远点
         * @param thFarPoints 远点距离阈值
         * @return 匹配数量
         * 
         * 用途：跟踪局部地图 (Tracking线程)
         * 将局部地图点投影到当前帧，在投影点周围搜索匹配特征点
         */
        int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, float fNNratio, const float th=3, const bool bFarPoints = false, const float thFarPoints = 50.0f);

        /**
         * @brief 将上一帧跟踪的地图点投影到当前帧并搜索匹配
         * @param CurrentFrame 当前帧
         * @param LastFrame 上一帧
         * @param th 搜索半径
         * @param bMono 是否为单目模式
         * @return 匹配数量
         * 
         * 用途：从上一帧跟踪 (Tracking线程)
         * 基于恒定运动模型的假设，在相邻帧间进行匹配
         */
        int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

        /**
         * @brief 将关键帧中看到的地图点投影到帧中并搜索匹配
         * @param CurrentFrame 当前帧
         * @param pKF 关键帧
         * @param sAlreadyFound 已找到的地图点集合
         * @param th 搜索半径
         * @param threshold 距离阈值
         * @return 匹配数量
         * 
         * 用途：重定位 (Tracking线程)
         * 在重定位时，使用候选关键帧的地图点来匹配当前帧
         */
        int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const float threshold);

        /**
         * @brief 使用相似变换投影地图点并搜索匹配
         * @param pKF 关键帧
         * @param Scw 相似变换矩阵
         * @param vpPoints 地图点列表
         * @param vpMatched 输出匹配的地图点
         * @param th 搜索半径
         * @param threshold 距离阈值
         * @return 匹配数量
         * 
         * 用途：回环检测 (Loop Closing线程)
         * 在回环检测时，使用Sim3变换将候选回环帧的地图点投影到当前帧
         */
        int SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th, const float threshold);

        /**
         * @brief 使用相似变换投影地图点并搜索匹配（增强版）
         * @param pKF 关键帧
         * @param Scw 相似变换矩阵
         * @param vpPoints 地图点列表
         * @param vpPointsKFs 地图点对应的关键帧列表
         * @param vpMatched 输出匹配的地图点
         * @param vpMatchedKF 输出匹配的关键帧
         * @param th 搜索半径
         * @param threshold 距离阈值
         * @return 匹配数量
         * 
         * 用途：位置识别 (回环检测和地图合并)
         * 相比上一个版本，额外记录了匹配点对应的关键帧信息
         */
        int SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs, std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, const float threshold);

        // ==================== 词袋匹配方法族 ====================

        /**
         * @brief 在关键帧和帧之间使用词袋模型搜索匹配
         * @param pKF 关键帧
         * @param F 当前帧
         * @param vpMapPointMatches 输出匹配的地图点
         * @return 匹配数量
         * 
         * 用途：重定位和回环检测
         * 通过词袋模型加速匹配，只比较在同一视觉单词中的特征
         */
        int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
        
        /**
         * @brief 在两个关键帧之间使用词袋模型搜索匹配
         * @param pKF1 关键帧1
         * @param pKF2 关键帧2
         * @param vpMatches12 输出匹配关系
         * @return 匹配数量
         * 
         * 用途：关键帧间数据关联
         * 用于建立关键帧间的匹配关系，支持局部建图和回环检测
         */
        int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

        // ==================== 特殊场景匹配方法 ====================

        /**
         * @brief 地图初始化的特征匹配（仅单目 case）
         * @param F1 第一帧
         * @param F2 第二帧
         * @param vbPrevMatched 先前匹配点位置
         * @param vnMatches12 输出匹配关系
         * @param fNNratio 最近邻比率
         * @param windowSize 搜索窗口大小
         * @return 匹配数量
         * 
         * 用途：单目SLAM初始化
         * 在两帧间寻找足够的特征匹配来初始化地图
         */
        int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, float fNNratio, int windowSize=10);

        /**
         * @brief 三角化新地图点的匹配，检查极线约束
         * @param pKF1 关键帧1
         * @param pKF2 关键帧2
         * @param vMatchedPairs 输出匹配对
         * @param bOnlyStereo 是否仅使用立体信息
         * @param bCoarse 是否粗匹配
         * @param vfDebugInfo 调试信息
         * @return 匹配数量
         * 
         * 用途：创建新地图点
         * 在两关键帧间寻找可以三角化的特征匹配，用于扩展地图
         */
        int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2,
                                   std::vector<std::pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse = false, std::vector<float> *vfDebugInfo = nullptr) const;

        /**
         * @brief 在Sim3变换下搜索两个关键帧间的匹配
         * @param pKF1 关键帧1
         * @param pKF2 关键帧2
         * @param vpMatches12 输出匹配关系
         * @param S12 Sim3变换
         * @param th 距离阈值
         * @return 匹配数量
         * 
         * 用途：回环检测和地图合并
         * 考虑尺度变化的匹配，用于精确的回环校正
         */
        int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th);

        // ==================== 地图点融合方法 ====================

        /**
         * @brief 将地图点投影到关键帧并搜索重复的地图点进行融合
         * @param pKF 关键帧
         * @param vpMapPoints 待融合的地图点列表
         * @param th 搜索半径
         * @param bRight 是否右目相机
         * @return 融合数量
         * 
         * 用途：地图点去冗余
         * 将相似的地图点融合，减少地图冗余，提高一致性
         */
        int Fuse(KeyFrame* pKF, const std::vector<MapPoint *> &vpMapPoints, const float th=3.0, const bool bRight = false);

        /**
         * @brief 使用给定Sim3变换将地图点投影到关键帧并融合重复点
         * @param pKF 关键帧
         * @param Scw 相似变换矩阵
         * @param vpPoints 地图点列表
         * @param th 搜索半径
         * @param vpReplacePoint 输出被替换的地图点
         * @return 融合数量
         * 
         * 用途：回环校正时的地图点融合
         * 在回环校正后，将校正前后的地图点进行融合
         */
        int Fuse(KeyFrame* pKF, Sophus::Sim3f &Scw, const std::vector<MapPoint*> &vpPoints, float th, std::vector<MapPoint *> &vpReplacePoint);

        /**
         * @brief 根据响应值过滤匹配
         * @param pKF 关键帧
         * @param vpMatches 匹配关系
         * @param fGoodRes 好的响应值阈值
         * @param nMinMatches 最小匹配数量
         * @return 过滤后的匹配数量
         * 
         * 用途：匹配质量优化
         * 根据特征点的响应值筛选高质量的匹配
         */
        int FilterMatchesByResponces(const KeyFrame* pKF, std::vector<MapPoint*>& vpMatches, float fGoodRes, int nMinMatches = -1);

    public:
        // ==================== 静态常量 ====================
        static const float TH_LOW;        // 低距离阈值（宽松匹配）
        static const float TH_HIGH;       // 高距离阈值（严格匹配）
        static const int HISTO_LENGTH;    // 直方图长度（用于方向一致性检查）
        
        // Eigen内存对齐宏
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
        // ==================== 保护成员方法 ====================
        
        /**
         * @brief 根据视角余弦计算搜索半径
         * @param viewCos 视角余弦值
         * @return 搜索半径
         * 
         * 视角越正对（余弦值越大），搜索半径越小
         */
        float RadiusByViewingCos(const float &viewCos);

        /**
         * @brief 计算直方图中的三个最大值
         * @param histo 直方图数据
         * @param L 直方图长度
         * @param ind1 最大值索引1（输出）
         * @param ind2 最大值索引2（输出）
         * @param ind3 最大值索引3（输出）
         * 
         * 用途：方向一致性检查
         * 用于筛选主方向一致的特征匹配，提高匹配质量
         */
        void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    };
} // namespace ORB_SLAM3
#endif // ORBMATCHER_H
