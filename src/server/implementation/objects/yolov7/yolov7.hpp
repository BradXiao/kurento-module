
#pragma once
#include "utils.hpp"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

class Logger;

class Yolov7trt {

public:
  /// @brief the official pre-trained model classes. (COCO Dataset)
  const static std::vector<std::string> CLASSNAMES;

  Yolov7trt(const std::string &modelPath, const int &device, std::string name);
  ~Yolov7trt();
  
  void infer(const cv::Mat &rgbImg, std::vector<utils::Obj> &output);

private:
  int deviceID;
  Logger *gLogger = nullptr;
  nvinfer1::IRuntime *runtime = nullptr;
  nvinfer1::ICudaEngine *engine = nullptr;
  nvinfer1::IExecutionContext *context = nullptr;
  cudaStream_t stream;
  utils::EngineIO engineIO;

  void initModel(const std::string &modelPath);
  void initEngineIO();
  void initEngineIO(bool allocateMem);
};