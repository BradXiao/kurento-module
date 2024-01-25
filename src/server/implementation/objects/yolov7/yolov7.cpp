
#include "yolov7.hpp"
#include "utils.hpp"
#include <NvInferPlugin.h>
#include <filesystem>
#include <fstream>
#include <gst/gst.h>
#include <stdexcept>

GST_DEBUG_CATEGORY_STATIC(obj_det_yolov7);
#define GST_CAT_DEFAULT obj_det_yolov7

namespace fs = std::filesystem;

static const int inputChannel = 3;
static const int inputWH = 640;

const std::vector<std::string> Yolov7trt::CLASSNAMES = {
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

class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity, const nvinfer1::AsciiChar *msg) noexcept override {
    switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
      GST_ERROR("tensorrt: %s", msg);
      break;
    case nvinfer1::ILogger::Severity::kERROR:
      GST_ERROR("tensorrt: %s", msg);
      break;
    case nvinfer1::ILogger::Severity::kWARNING:
      GST_WARNING("tensorrt: %s", msg);
      break;
    case nvinfer1::ILogger::Severity::kINFO:
      GST_INFO("tensorrt: %s", msg);
      break;
    default:
      GST_LOG("tensorrt: %s", msg);
      break;
    }
  }
};

Yolov7trt::Yolov7trt(const std::string &modelPath, const int &device, std::string name) : deviceID(device) {
  GST_DEBUG_CATEGORY_INIT(obj_det_yolov7, (std::string("ObjDetYolov7-") + name).c_str(), GST_DEBUG_BG_GREEN, "ObjDetYolov7");
  // cuda device check
  int cudaCount = -1;
  cudaGetDeviceCount(&cudaCount);
  GST_INFO("found %d cuda devices and set device to %d", cudaCount, device);
  if (cudaCount <= 0) {
    GST_ERROR("no gpu found, cannot init model");
    throw std::runtime_error("no gpu found");
  }
  if (device < cudaCount - 1) {
    GST_ERROR("incorrect cuda device configuration");
    throw std::runtime_error("incorrect cuda device configuration");
  }

  // load model
  GST_INFO("init model");
  this->initModel(modelPath);

  // define input and output
  GST_INFO("allocate memory");
  initEngineIO();

  // warmup
  GST_INFO("warmup");
  for (int i = 0; i < 10; i++) {
    GST_DEBUG("warmup %d", i);
    cv::Mat dummyImg(640, 640, CV_8UC3, cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    std::vector<utils::Obj> dummyObjs;
    this->infer(dummyImg, dummyObjs);
  }
};

void Yolov7trt::infer(const cv::Mat &rgbImg, std::vector<utils::Obj> &output) {
  utils::Yolov7Input input;
  GST_DEBUG("preprocess");
  utils::preprocess(rgbImg, input);
  GST_DEBUG("copy input to gpu(async)");
  cudaMemcpyAsync(this->engineIO.inputBufferGPU[0], input.mat.ptr<float>(), input.mat.total() * input.mat.elemSize(),
                  cudaMemcpyHostToDevice, this->stream);

  GST_DEBUG("infer(enqueue)");
  this->context->enqueueV2(this->engineIO.combinedBuffersGPU.data(), this->stream, nullptr); // TODO: deprecated

  GST_DEBUG("copy output to cpu(async)");
  int totalOutput = static_cast<int>(this->engineIO.outputBindings.size());
  for (int i = 0; i < totalOutput; i++) {
    size_t size = this->engineIO.outputBindings[i].size * this->engineIO.outputBindings[i].dataSize;
    cudaMemcpyAsync(this->engineIO.outputBuffersCPU[i], this->engineIO.outputBuffersGPU[i], size, cudaMemcpyDeviceToHost,
                    this->stream);
  }

  GST_DEBUG("cuda stream sync");
  cudaStreamSynchronize(this->stream);
  GST_DEBUG("postprocess");
  utils::postprocess(this->engineIO.outputBuffersCPU, input, output, this->CLASSNAMES);
};

void Yolov7trt::initModel(const std::string &modelPath) {
  GST_INFO("set device %d", this->deviceID);
  cudaSetDevice(this->deviceID);
  // runtime
  GST_INFO("init logger");
  this->gLogger = new Logger();
  initLibNvInferPlugins(gLogger, "");

  GST_INFO("create infer runtime");
  this->runtime = nvinfer1::createInferRuntime(*gLogger);
  assert(this->runtime != nullptr);

  // engine
  GST_INFO("check model file");
  if (fs::exists(modelPath) == false) {
    GST_ERROR("model not found: %s", modelPath.c_str());
    throw std::runtime_error("model not found: " + modelPath);
  }
  std::ifstream file(modelPath, std::ios::binary);
  if (file.good() == false) {
    GST_ERROR("model cannot load: %s", modelPath.c_str());
    throw std::runtime_error("model cannot load: " + modelPath);
  }
  GST_INFO("load model");
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  char *buffer = new char[size];
  file.read(buffer, size);
  this->engine = this->runtime->deserializeCudaEngine(buffer, size);
  delete[] buffer;
  assert(this->engine != nullptr);

  // context
  GST_INFO("create execution context");
  this->context = this->engine->createExecutionContext();
  assert(this->context != nullptr);

  GST_INFO("create stream");
  cudaStreamCreate(&stream);
};

void Yolov7trt::initEngineIO() { this->initEngineIO(true); };

void Yolov7trt::initEngineIO(bool allocateMem) {
  GST_INFO("get binging numbers");
  int bindingCount = this->engine->getNbBindings(); // TODO: deprecated
  assert(bindingCount == 5);
  this->engineIO.bindingCount = bindingCount;

  // input
  GST_INFO("get input binding info");
  assert(this->engine->bindingIsInput(0)); // TODO: deprecated
  utils::getBindingInfo(this->engineIO.inputBinding, this->engine, 0);

  GST_INFO("set binding dim= %dx%dx%dx%d", 1, inputChannel, inputWH, inputWH);
  this->context->setBindingDimensions(0, nvinfer1::Dims4{1, inputChannel, inputWH, inputWH}); // TODO: deprecated

  // output
  GST_INFO("get output binding info");
  for (int i = 1; i < bindingCount; i++) {
    assert(this->engine->bindingIsInput(i) == false);
    utils::BindingInfo info;
    utils::getBindingInfo(info, this->engine, i);
    this->engineIO.outputBindings.push_back(info);
  }

  if (allocateMem == true) {
    // input (gpu only)
    GST_INFO("allocate input buffer");
    void *inputBuffer;
    cudaMalloc(&inputBuffer, this->engineIO.inputBinding.size * this->engineIO.inputBinding.dataSize);
    this->engineIO.inputBufferGPU.push_back(inputBuffer);

    // output
    GST_INFO("allocate output buffer");
    for (auto &binding : this->engineIO.outputBindings) {
      assert(binding.isInput == false);
      // gpu
      void *outputBufferGPU;
      cudaMalloc(&outputBufferGPU, binding.size * binding.dataSize);
      this->engineIO.outputBuffersGPU.push_back(outputBufferGPU);
      // cpu
      void *outputBufferCPU;
      cudaHostAlloc(&outputBufferCPU, binding.size * binding.dataSize, 0);
      this->engineIO.outputBuffersCPU.push_back(outputBufferCPU);
    }

    // a combined pointer for model input
    this->engineIO.combinedBuffersGPU.reserve(this->engineIO.inputBufferGPU.size() + this->engineIO.outputBuffersGPU.size());
    this->engineIO.combinedBuffersGPU.insert(this->engineIO.combinedBuffersGPU.end(), this->engineIO.inputBufferGPU.begin(),
                                             this->engineIO.inputBufferGPU.end());
    this->engineIO.combinedBuffersGPU.insert(this->engineIO.combinedBuffersGPU.end(), this->engineIO.outputBuffersGPU.begin(),
                                             this->engineIO.outputBuffersGPU.end());
  } else {
    GST_INFO("skip allocating memory");
  }
};

Yolov7trt::~Yolov7trt() {
  GST_INFO("destroy model");
  this->context->destroy(); // TODO: deprecated
  this->engine->destroy();  // TODO: deprecated
  this->runtime->destroy(); // TODO: deprecated
  cudaStreamDestroy(this->stream);
  for (auto &ptr : this->engineIO.combinedBuffersGPU) {
    cudaFree(ptr);
  }
  for (auto &ptr : this->engineIO.outputBuffersCPU) {
    cudaFreeHost(ptr);
  }

  delete this->gLogger;
};