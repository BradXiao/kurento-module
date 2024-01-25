#pragma once
#include <NvInfer.h>

#include <opencv2/opencv.hpp>

#include "yolov7.hpp"
#include <json/json.h>

namespace utils {

/// @brief Model input or output binding info
struct BindingInfo {
  int dataSize;
  size_t size = 1;
  nvinfer1::Dims dims;
  std::string name;
  bool isInput;
};

struct Yolov7Input {
  cv::Mat mat;
  cv::Size inputSize;
  float ratio;
  int dw;
  int dh;
};

struct Obj {
  cv::Point p1;
  cv::Point p2;
  std::string name;
  int classIdx;
  float confi;

  bool operator<(const Obj &other) const { return confi > other.confi; }
};

///@brief Struct contains binding info and memory locations
struct EngineIO {
  int bindingCount;
  utils::BindingInfo inputBinding;
  std::vector<utils::BindingInfo> outputBindings;
  std::vector<void *> inputBufferGPU;
  std::vector<void *> outputBuffersGPU;
  std::vector<void *> outputBuffersCPU;
  std::vector<void *> combinedBuffersGPU;
};

/**
 * @brief Get the data type size for memory allocation
 *
 * @param dataType
 * @return int
 */
static inline int getDataTypeSize(const nvinfer1::DataType &dataType) {
  switch (dataType) {
  case nvinfer1::DataType::kFLOAT:
    return sizeof(float);
  case nvinfer1::DataType::kINT32:
    return sizeof(int);
  default:
    throw std::runtime_error("data type is not supported");
  }
};

/**
 * @brief Get the binding info of a specific layer
 *
 * @param engine
 * @param index
 * @return BindingInfo
 */
static inline void getBindingInfo(BindingInfo &info, const nvinfer1::ICudaEngine *engine, int index) {
  std::string name = engine->getBindingName(index);          // TODO: deprecated
  nvinfer1::Dims dims = engine->getBindingDimensions(index); // TODO: deprecated
  int size = 1;
  for (int i = 0; i < dims.nbDims; i++) {
    size *= dims.d[i];
  }

  info.dataSize = getDataTypeSize(engine->getBindingDataType(index)); // TODO: deprecated
  info.size = static_cast<size_t>(size);
  info.dims = dims;
  info.name = name;
  info.isInput = engine->bindingIsInput(index); // TODO: deprecated
};

/**
 * @brief Image preprocess (letterbox and normalization)
 *
 * @param rgbImg
 * @param input
 * @param wh target width/height
 * @param padColor
 */
static inline void preprocess(const cv::Mat &rgbImg, Yolov7Input &input, int wh, int padColor) {

  if (rgbImg.channels() == 4) {
    cv::cvtColor(rgbImg, input.mat, cv::COLOR_RGBA2RGB);
  } else {
    input.mat = rgbImg.clone();
  }
  cv::Size shape = rgbImg.size();
  float targetWidth = wh;
  float targetHeight = wh;
  float r = std::min(targetWidth / shape.width, targetHeight / shape.height);
  input.inputSize = shape;

  int newUnpadWidth = std::lround(shape.width * r);
  int newUnpadHeight = std::lround(shape.height * r);

  newUnpadWidth = std::min(newUnpadWidth, wh);
  newUnpadHeight = std::min(newUnpadHeight, wh);

  float dw = (wh - newUnpadWidth) / 2.;
  float dh = (wh - newUnpadHeight) / 2.;

  if (shape.width != newUnpadWidth || shape.height != newUnpadHeight) {
    cv::resize(input.mat, input.mat, cv::Size(newUnpadWidth, newUnpadHeight));
  }
  int top = std::lround(dh - 0.1);
  int bottom = std::lround(dh + 0.1);
  int left = std::lround(dw - 0.1);
  int right = std::lround(dw + 0.1);

  cv::copyMakeBorder(input.mat, input.mat, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(padColor, padColor, padColor));
  cv::dnn::blobFromImage(input.mat, input.mat, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), false, false, CV_32F);

  input.ratio = r;
  input.dw = std::lround(dw);
  input.dh = std::lround(dh);
};

/**
 * @brief Image preprocess (letterbox and normalization)
 *
 * @param rgbImg
 * @param input
 */
static inline void preprocess(const cv::Mat &rgbImg, Yolov7Input &input) { preprocess(rgbImg, input, 640, 114); };

/**
 * @brief Model output postprocess (convert to Obj class)
 *
 * @param outputBuffer
 * @param input
 * @param objs
 */
static inline void postprocess(const std::vector<void *> &outputBuffer, const Yolov7Input &input, std::vector<Obj> &objs,
                               const std::vector<std::string> &CLASSNAMES) {
  objs.clear();
  const int *boxCount = static_cast<int *>(outputBuffer[0]);
  const float *boxes = static_cast<float *>(outputBuffer[1]);
  const float *confidences = static_cast<float *>(outputBuffer[2]);
  const int *labels = static_cast<int *>(outputBuffer[3]);
  for (int i = 0; i < boxCount[0]; i++) {
    const float *box = boxes + i * 4;

    int x1 = std::min(std::max((int)std::lround((*box++ - input.dw) / input.ratio), 0), input.inputSize.width);
    int y1 = std::min(std::max((int)std::lround((*box++ - input.dh) / input.ratio), 0), input.inputSize.height);
    int x2 = std::min(std::max((int)std::lround((*box++ - input.dw) / input.ratio), 0), input.inputSize.width);
    int y2 = std::min(std::max((int)std::lround((*box++ - input.dh) / input.ratio), 0), input.inputSize.height);

    Obj obj;
    obj.p1 = cv::Point(x1, y1);
    obj.p2 = cv::Point(x2, y2);
    obj.confi = *(confidences + i);
    obj.classIdx = *(labels + i);
    obj.name = CLASSNAMES[obj.classIdx];
    objs.push_back(obj);
  }
};

/**
 * @brief Draw boxes
 *
 * @param srcRGBImg
 * @param desRGBImg
 * @param objs
 * @param swapBR
 * @param frontScale
 */
static inline void drawObjs(const cv::Mat &srcRGBImg, cv::Mat &desRGBImg, const std::vector<Obj> &objs, bool swapBR, float frontScale,
                            const cv::Scalar &color) {
  if (&srcRGBImg != &desRGBImg) {
    desRGBImg = srcRGBImg.clone();
  }

  if (swapBR == true) {
    cv::cvtColor(desRGBImg, desRGBImg, cv::COLOR_RGB2BGR);
  }

  for (const Obj &obj : objs) {
    if (color[0] == -1) {
      cv::rectangle(desRGBImg, obj.p1, obj.p2, cv::Scalar(rand() % 256, rand() % 256, rand() % 256), 2);
    } else {
      cv::rectangle(desRGBImg, obj.p1, obj.p2, color, 2);
    }

    std::stringstream ss;
    ss << obj.name << "-" << std::fixed << std::setprecision(2) << obj.confi;
    std::string label = ss.str();
    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, frontScale, 1, &baseLine);

    cv::rectangle(desRGBImg, cv::Rect(obj.p1.x, obj.p1.y + 1, labelSize.width, labelSize.height + baseLine), cv::Scalar(0, 255, 0),
                  -1);

    cv::putText(desRGBImg, label, cv::Point(obj.p1.x, obj.p1.y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, frontScale,
                cv::Scalar(0, 0, 0), 1);
  }
};

static inline void drawObjsFixedColor(const cv::Mat &srcRGBImg, cv::Mat &desRGBImg, const std::vector<Obj> &objs, bool swapBR,
                                      float frontScale, const std::vector<cv::Scalar> &colors) {
  if (&srcRGBImg != &desRGBImg) {
    desRGBImg = srcRGBImg.clone();
  }

  if (swapBR == true) {
    cv::cvtColor(desRGBImg, desRGBImg, cv::COLOR_RGB2BGR);
  }

  for (const Obj &obj : objs) {
    cv::rectangle(desRGBImg, obj.p1, obj.p2, colors[obj.classIdx], 2);

    std::stringstream ss;
    ss << obj.name << "-" << std::fixed << std::setprecision(2) << obj.confi;
    std::string label = ss.str();
    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, frontScale, 1, &baseLine);

    cv::rectangle(desRGBImg, cv::Rect(obj.p1.x, obj.p1.y + 1, labelSize.width, labelSize.height + baseLine), cv::Scalar(0, 255, 0),
                  -1);

    cv::putText(desRGBImg, label, cv::Point(obj.p1.x, obj.p1.y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, frontScale,
                cv::Scalar(0, 0, 0), 1);
  }
}

/**
 * @brief Draw boxes
 *
 * @param srcRGBImg
 * @param desRGBImg
 * @param objs
 */
static inline void drawObjs(const cv::Mat &srcRGBImg, cv::Mat &desRGBImg, const std::vector<Obj> &objs) {
  drawObjs(srcRGBImg, desRGBImg, objs, false, 0.4, cv::Scalar(-1, -1, -1));
};

/**
 * @brief Convert json value to string
 *
 * @param input
 * @return std::string
 */
static inline std::string jsonToString(Json::Value &input) {
  Json::FastWriter writer;
  return writer.write(input);
};

} // namespace utils
