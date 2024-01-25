#include "ModelPool.hpp"
#include "utils.hpp"
#include "yolov7.hpp"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gst/gst.h>
#include <json/json.h>
#include <mutex>

namespace fs = std::filesystem;

GST_DEBUG_CATEGORY_STATIC(obj_det_model_pool);
#define GST_CAT_DEFAULT obj_det_model_pool

namespace kurento {
namespace module {
namespace objdet {

ModelBundle::~ModelBundle() {
  for (Yolov7trt *model : this->models) {
    delete model;
  }
}

ModelPool::ModelPool() {

  GST_DEBUG_CATEGORY_INIT(obj_det_model_pool, "ObjDetModelPool", GST_DEBUG_BG_YELLOW, "ObjDetModelPool");
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  GST_DEBUG("init");

  // read config
  GST_INFO("check config file");
  const char *path = std::getenv("OBJDET_CONFIG");
  if (path == nullptr) {
    GST_ERROR("OBJDET_CONFIG is not specified");
    throw std::runtime_error("environment variable OBJDET_CONFIG is not specified");
  }

  if (fs::exists(path) == false) {
    GST_ERROR("object detection config file not found: %s", path);
    throw std::runtime_error(std::string("object detection config file not found: ") + path);
  }

  std::ifstream fileStream(path, std::ifstream::binary);
  if (fileStream.good() == false) {
    GST_ERROR("object detection config file cannot load: %s", path);
    throw std::runtime_error(std::string("object detection config file cannot load: ") + path);
  }

  GST_INFO("read config file");
  Json::Value config;
  Json::Reader reader;
  if (reader.parse(fileStream, config) == false) {
    GST_ERROR("object detection config file JSON format error: %s", path);
    throw std::runtime_error(std::string("object detection config file JSON format error: ") + path);
  }

  GST_INFO("finish loading config file");
  GST_INFO("%s", utils::jsonToString(config).c_str());

  int deviceId = std::max(config["device_id"].asInt(), 0);
  cudaSetDevice(deviceId);
  GST_INFO("device id = %d", deviceId);
  // init models
  this->defaultModelName = config["default_model_name"].asString();
  int totalModelNum = static_cast<int>(config["models"].size());
  for (int i = 0; i < totalModelNum; i++) {
    Json::Value modelParam = config["models"][i];
    // check model enabled
    if (modelParam["enabled"].asBool() == false) {
      GST_INFO("Skip disabled model %s", modelParam["name"].asString().c_str());
      continue;
    }
    if (modelParam["name"].asString() == "default") {
      GST_ERROR("model name 'default' is a pre-defined keyword");
      throw std::runtime_error("model name 'default' is a pre-defined keyword");
      break;
    }

    int maxModelLimit = std::max(modelParam["max_model_limit"].asInt(), 1);
    GST_INFO("Start init %d %s models", maxModelLimit, modelParam["name"].asString().c_str());

    // check model file
    std::string modelPath = modelParam["model_abs_path"].asString();
    GST_INFO("check model file");
    if (fs::exists(modelPath) == false) {
      GST_ERROR("object detection model not found: %s", modelPath.c_str());
      throw std::runtime_error(std::string("object detection model not found: ") + modelPath);
    }
    // load models
    ModelBundle *bundle = new ModelBundle();
    for (int i = 0; i < maxModelLimit; i++) {
      this->checkVRAM(deviceId, 500000000);
      GST_INFO("Init %d/%d %s model", i + 1, maxModelLimit, modelParam["name"].asString().c_str());
      Yolov7trt *md;
      try {
        md = new Yolov7trt(modelPath, deviceId, std::to_string(i));
        GST_INFO("Finish init %d/%d %s model", i + 1, maxModelLimit, modelParam["name"].asString().c_str());
      } catch (const std::exception &e) {
        GST_ERROR("Error init %d/%d %s model", i + 1, maxModelLimit, modelParam["name"].asString().c_str());
        continue;
      }
      bundle->models.push_back(md);
      uintptr_t modelAddress = reinterpret_cast<uintptr_t>(md);
      bundle->isUsed[modelAddress] = false;
      GST_INFO("Added %d/%d %s model", i + 1, maxModelLimit, modelParam["name"].asString().c_str());
    }

    this->modelBundles[modelParam["name"].asString()] = bundle;
  }

  if (this->modelBundles.find(this->defaultModelName) == this->modelBundles.end()) {
    GST_ERROR("default model name not found (%s)", this->defaultModelName.c_str());
    throw std::runtime_error("default model name not found");
  }
}

int ModelPool::getAvailableCount(const std::string &modelName) {
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  if (this->modelExists(modelName) == false) {
    GST_ERROR("model bundle %s not found", modelName.c_str());
    return -1;
  }
  int count = 0;
  for (auto const &[_, used] : this->modelBundles[modelName]->isUsed) {
    count += used ? 1 : 0;
  }
  GST_DEBUG("available %s model is %d", modelName.c_str(), count);
  return count;
}

bool ModelPool::isAvailable(const std::string &modelName) {
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  if (this->modelExists(modelName) == false) {
    GST_ERROR("model %s not found", modelName.c_str());
    return false;
  }
  for (auto const &[_, used] : this->modelBundles[modelName]->isUsed) {
    if (used == false) {
      GST_DEBUG("is available=true");
      return true;
    }
  }
  GST_DEBUG("is available=false");
  return false;
}

Yolov7trt *ModelPool::getModel(const std::string &modelName) {
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  if (this->modelExists(modelName) == false) {
    GST_ERROR("model %s not found", modelName.c_str());
    return nullptr;
  }
  GST_INFO("get a %s model", modelName.c_str());
  ModelBundle *bundle = this->modelBundles[modelName];
  for (Yolov7trt *model : bundle->models) {
    uintptr_t address = reinterpret_cast<uintptr_t>(model);
    if (bundle->isUsed[address] == false) {
      bundle->isUsed[address] = true;
      GST_INFO("get a model successfully");
      return model;
    }
  }

  // take timeout model
  GST_DEBUG("try get a timeout model");
  bool isModelReleased = this->updateSession(modelName);
  if (isModelReleased) {
    return this->getModel(modelName);
  }
  GST_WARNING("try to get %s model but no model is available", modelName.c_str());
  return nullptr;
}

std::string ModelPool::getDefaultModelName() {
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  return this->defaultModelName;
}

void ModelPool::returnModel(const std::string &modelName, Yolov7trt *model, const std::string &sessionId) {
  GST_INFO("return a %s model", modelName.c_str());
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  if (this->modelExists(modelName) == false) {
    GST_ERROR("model %s not found", modelName.c_str());
    return;
  }
  uintptr_t address = reinterpret_cast<uintptr_t>(model);
  ModelBundle *bundle = this->modelBundles[modelName];
  bundle->isUsed[address] = false;
  bundle->sessionHeartbeat.erase(sessionId);
  bundle->sessionToModel.erase(sessionId);
}

void ModelPool::getModelNames(std::vector<std::string> &names) {
  GST_INFO("get model names");
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  names.clear();
  for (auto &[modelName, _] : this->modelBundles) {
    names.push_back(modelName);
  }
}

bool ModelPool::modelExists(const std::string &modelName) {
  GST_INFO("check model exists");
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  return this->modelBundles.find(modelName) != this->modelBundles.end();
}

void ModelPool::registerSession(const std::string &modelName, Yolov7trt *model, const std::string &sessionId) {
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  if (this->modelExists(modelName) == false) {
    GST_ERROR("model %s not found", modelName.c_str());
    return;
  }
  GST_INFO("get a session %s", sessionId.c_str());
  ModelBundle *bundle = this->modelBundles[modelName];
  bundle->sessionHeartbeat[sessionId] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  bundle->sessionToModel[sessionId] = model;
}

void ModelPool::heartbeat(const std::string &modelName, std::string sessionId) {
  GST_DEBUG("heartbeat %s", sessionId.c_str());
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  if (this->modelExists(modelName) == false) {
    GST_ERROR("model %s not found", modelName.c_str());
    return;
  }
  ModelBundle *bundle = this->modelBundles[modelName];
  if (bundle->sessionHeartbeat.find(sessionId) == bundle->sessionHeartbeat.end()) {
    return;
  }
  bundle->sessionHeartbeat[sessionId] = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
}

bool ModelPool::sessionExists(const std::string &modelName, std::string sessionId) {
  GST_DEBUG("check session exists %s", sessionId.c_str());
  std::lock_guard<std::recursive_mutex> lockNow(this->lock);
  if (this->modelExists(modelName) == false) {
    GST_ERROR("model %s not found", modelName.c_str());
    return false;
  }
  this->updateSession(modelName);
  ModelBundle *bundle = this->modelBundles[modelName];
  return bundle->sessionHeartbeat.find(sessionId) != bundle->sessionHeartbeat.end();
}

ModelPool::~ModelPool() {
  GST_INFO("destroy models");
  for (auto &[_, modelBundle] : this->modelBundles) {
    delete modelBundle;
  }
}

bool ModelPool::updateSession(const std::string &modelName) {
  if (this->modelExists(modelName) == false) {
    GST_ERROR("model %s not found", modelName.c_str());
    return false;
  }
  ModelBundle *bundle = this->modelBundles[modelName];
  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::vector<std::string> toBeDelete;
  for (const auto &[id, timestamp] : bundle->sessionHeartbeat) {
    if (now - timestamp > 60) {
      toBeDelete.push_back(id);
    }
  }

  for (const std::string &id : toBeDelete) {
    GST_DEBUG("release expired model resourse %s", id.c_str());
    bundle->sessionHeartbeat.erase(id);
    uintptr_t address = reinterpret_cast<uintptr_t>(bundle->sessionToModel[id]);
    bundle->isUsed[address] = false;
    bundle->sessionToModel.erase(id);
  }
  return toBeDelete.size() != 0;
}

void ModelPool::checkVRAM(const int deviceId, const size_t minBytes) {
  cudaSetDevice(deviceId);
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  if (freeMem < minBytes) {
    GST_ERROR("GPU available memory insufficient %zu Bytes (<%zu), please disable or decrease the model number limit in config file",
              freeMem, minBytes);
    throw std::runtime_error("insufficient VRAM");
  }
}

} // namespace objdet
} // namespace module
} // namespace kurento