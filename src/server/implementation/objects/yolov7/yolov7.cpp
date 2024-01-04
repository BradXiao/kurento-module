
#include "yolov7.hpp"
#include "utils.hpp"
#include <NvInferPlugin.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

static const int inputChannel = 3;
static const int inputWH = 640;

class Logger : public nvinfer1::ILogger {
  void log(nvinfer1::ILogger::Severity severity, const nvinfer1::AsciiChar *msg) noexcept override {
    if (severity != nvinfer1::ILogger::Severity::kINFO) {
      std::cerr << msg << std::endl;
    }
  }
};

Yolov7trt::Yolov7trt(const std::string &modelPath, const int &device) : deviceID(device) {
  // cuda device check
  int cudaCount = -1;
  cudaGetDeviceCount(&cudaCount);
  if (cudaCount <= 0) {
    throw std::runtime_error("no gpu found");
  }
  if (device < cudaCount - 1) {
    throw std::runtime_error("incorrect cuda device number");
  }

  // load model
  this->initModel(modelPath);

  // define input and output
  initEngineIO();

  // warmup
  for (int i = 0; i < 10; i++) {
    cv::Mat dummpyImg(640, 640, CV_8UC3, cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    std::vector<utils::Obj> dummyObjs;
    this->infer(dummpyImg, dummyObjs);
  }
};

void Yolov7trt::infer(const cv::Mat &rgbImg, std::vector<utils::Obj> &output) {
  utils::Yolov7Input input;
  utils::preprocess(rgbImg, input);
  cudaMemcpyAsync(this->engineIO.inputBufferGPU[0], input.mat.ptr<float>(), input.mat.total() * input.mat.elemSize(),
                  cudaMemcpyHostToDevice, this->stream);

  // todo: deprecated API
  this->context->enqueueV2(this->engineIO.combinedBuffersGPU.data(), this->stream, nullptr);

  for (int i = 0; i < this->engineIO.outputBindings.size(); i++) {
    size_t size = this->engineIO.outputBindings[i].size * this->engineIO.outputBindings[i].dataSize;
    cudaMemcpyAsync(this->engineIO.outputBuffersCPU[i], this->engineIO.outputBuffersGPU[i], size, cudaMemcpyDeviceToHost,
                    this->stream);
  }

  cudaStreamSynchronize(this->stream);
  utils::postprocess(this->engineIO.outputBuffersCPU, input, output);
};

void Yolov7trt::initModel(const std::string &modelPath) {
  cudaSetDevice(this->deviceID);
  // runtime
  this->gLogger = new Logger();
  initLibNvInferPlugins(gLogger, "");

  this->runtime = nvinfer1::createInferRuntime(*gLogger);
  assert(this->runtime != nullptr);

  // engine
  if (fs::exists(modelPath) == false) {
    throw std::runtime_error("model not found: " + modelPath);
  }
  std::ifstream file(modelPath, std::ios::binary);
  if (file.good() == false) {
    throw std::runtime_error("model cannot load: " + modelPath);
  }
  file.seekg(0, std::ios::end);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  char *buffer = new char[size];
  file.read(buffer, size);
  this->engine = this->runtime->deserializeCudaEngine(buffer, size);
  delete[] buffer;
  assert(this->engine != nullptr);

  // context
  this->context = this->engine->createExecutionContext();
  assert(this->context != nullptr);

  cudaStreamCreate(&stream);
};

void Yolov7trt::initEngineIO() { this->initEngineIO(true); };

void Yolov7trt::initEngineIO(bool allocateMem) {

  int bindingCount = this->engine->getNbBindings();
  assert(bindingCount == 5);
  this->engineIO.bindingCount = bindingCount;

  // input
  assert(this->engine->bindingIsInput(0));
  utils::getBindingInfo(this->engineIO.inputBinding, this->engine, 0);
  this->context->setBindingDimensions(0, nvinfer1::Dims4{1, inputChannel, inputWH, inputWH});

  // output
  for (int i = 1; i < bindingCount; i++) {
    assert(this->engine->bindingIsInput(i) == false);
    utils::BindingInfo info;
    utils::getBindingInfo(info, this->engine, i);
    this->engineIO.outputBindings.push_back(info);
  }

  if (allocateMem == true) {
    // input (gpu only)
    void *inputBuffer;
    cudaMalloc(&inputBuffer, this->engineIO.inputBinding.size * this->engineIO.inputBinding.dataSize);
    this->engineIO.inputBufferGPU.push_back(inputBuffer);

    // output
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
  }
};

Yolov7trt::~Yolov7trt() {

  this->context->destroy();
  this->engine->destroy();
  this->runtime->destroy();
  cudaStreamDestroy(this->stream);
  for (auto &ptr : this->engineIO.combinedBuffersGPU) {
    cudaFree(ptr);
  }
  for (auto &ptr : this->engineIO.outputBuffersCPU) {
    cudaFreeHost(ptr);
  }

  delete this->gLogger;
};