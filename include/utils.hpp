#pragma once
#include <NvInfer.h>

#include <opencv2/opencv.hpp>

#include "yolov7.hpp"
#include <json/json.h>

namespace utils {
/// @brief The official pre-trained model classes. (COCO Dataset)
const static std::vector<std::string> CLASSNAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const static std::vector<cv::Scalar> CLASSCOLORS = {
    cv::Scalar(226, 209, 70),  cv::Scalar(25, 171, 239),  cv::Scalar(127, 166, 66),  cv::Scalar(105, 197, 57),
    cv::Scalar(7, 86, 180),    cv::Scalar(199, 156, 97),  cv::Scalar(184, 47, 200),  cv::Scalar(111, 38, 130),
    cv::Scalar(0, 80, 6),      cv::Scalar(179, 32, 45),   cv::Scalar(219, 26, 91),   cv::Scalar(239, 202, 167),
    cv::Scalar(10, 17, 6),     cv::Scalar(137, 126, 41),  cv::Scalar(204, 161, 22),  cv::Scalar(86, 133, 25),
    cv::Scalar(121, 226, 66),  cv::Scalar(112, 177, 12),  cv::Scalar(158, 65, 67),   cv::Scalar(127, 34, 206),
    cv::Scalar(200, 79, 64),   cv::Scalar(199, 91, 18),   cv::Scalar(183, 188, 72),  cv::Scalar(225, 11, 165),
    cv::Scalar(51, 254, 208),  cv::Scalar(165, 98, 16),   cv::Scalar(233, 130, 50),  cv::Scalar(13, 237, 196),
    cv::Scalar(255, 23, 247),  cv::Scalar(138, 244, 170), cv::Scalar(130, 48, 3),    cv::Scalar(136, 92, 52),
    cv::Scalar(41, 134, 192),  cv::Scalar(20, 183, 77),   cv::Scalar(78, 238, 65),   cv::Scalar(232, 52, 239),
    cv::Scalar(15, 111, 76),   cv::Scalar(82, 25, 99),    cv::Scalar(194, 167, 83),  cv::Scalar(123, 219, 222),
    cv::Scalar(120, 146, 229), cv::Scalar(16, 89, 117),   cv::Scalar(95, 87, 193),   cv::Scalar(81, 165, 203),
    cv::Scalar(206, 35, 246),  cv::Scalar(110, 46, 249),  cv::Scalar(200, 8, 75),    cv::Scalar(107, 75, 148),
    cv::Scalar(16, 138, 251),  cv::Scalar(145, 163, 214), cv::Scalar(201, 98, 69),   cv::Scalar(255, 91, 170),
    cv::Scalar(190, 81, 142),  cv::Scalar(169, 214, 4),   cv::Scalar(30, 13, 58),    cv::Scalar(21, 215, 172),
    cv::Scalar(203, 253, 117), cv::Scalar(180, 166, 154), cv::Scalar(172, 11, 121),  cv::Scalar(173, 142, 64),
    cv::Scalar(180, 113, 93),  cv::Scalar(85, 34, 109),   cv::Scalar(193, 221, 152), cv::Scalar(29, 78, 161),
    cv::Scalar(221, 166, 251), cv::Scalar(251, 218, 147), cv::Scalar(247, 42, 245),  cv::Scalar(93, 17, 240),
    cv::Scalar(65, 198, 85),   cv::Scalar(5, 240, 177),   cv::Scalar(204, 159, 188), cv::Scalar(15, 56, 101),
    cv::Scalar(82, 252, 30),   cv::Scalar(119, 212, 154), cv::Scalar(11, 221, 253),  cv::Scalar(186, 127, 100),
    cv::Scalar(156, 213, 158), cv::Scalar(153, 156, 177), cv::Scalar(86, 27, 124),   cv::Scalar(86, 18, 63)};

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
static int getDataTypeSize(const nvinfer1::DataType &dataType) {
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
static void getBindingInfo(BindingInfo &info, const nvinfer1::ICudaEngine *engine, int index) {
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
static void preprocess(const cv::Mat &rgbImg, Yolov7Input &input, int wh, int padColor) {

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
static void preprocess(const cv::Mat &rgbImg, Yolov7Input &input) { preprocess(rgbImg, input, 640, 114); };

/**
 * @brief Model output postproce (convert to Obj class)
 *
 * @param outputBuffer
 * @param input
 * @param objs
 */
static void postprocess(const std::vector<void *> &outputBuffer, const Yolov7Input &input, std::vector<Obj> &objs) {
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
static void drawObjs(const cv::Mat &srcRGBImg, cv::Mat &desRGBImg, const std::vector<Obj> &objs, bool swapBR, float frontScale) {
  if (&srcRGBImg != &desRGBImg) {
    desRGBImg = srcRGBImg.clone();
  }

  if (swapBR == true) {
    cv::cvtColor(desRGBImg, desRGBImg, cv::COLOR_RGB2BGR);
  }

  for (const Obj &obj : objs) {
    cv::rectangle(desRGBImg, obj.p1, obj.p2, cv::Scalar(rand() % 256, rand() % 256, rand() % 256), 2);

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

static void drawObjsFixedColor(const cv::Mat &srcRGBImg, cv::Mat &desRGBImg, const std::vector<Obj> &objs, bool swapBR,
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
static void drawObjs(const cv::Mat &srcRGBImg, cv::Mat &desRGBImg, const std::vector<Obj> &objs) {
  drawObjs(srcRGBImg, desRGBImg, objs, false, 0.4);
};

/**
 * @brief Convert json value to string
 *
 * @param input
 * @return std::string
 */
static std::string jsonToString(Json::Value &input) {
  Json::FastWriter writer;
  return writer.write(input);
};

} // namespace utils
