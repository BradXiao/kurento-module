#include "yolov7.hpp"

#include "utils.hpp"
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
class ModelPool {
public:
  ModelPool() {

    GST_DEBUG_CATEGORY_INIT(obj_det_model_pool, "ObjDetModelPool", GST_DEBUG_BG_YELLOW, "ObjDetModelPool");
    std::lock_guard<std::recursive_mutex> lockNow(lock);
    GST_DEBUG("init");

    // read config
    GST_INFO("check config file");
    const char *path = std::getenv("OBJDET_CONFIG");
    if (path == nullptr) {
      GST_ERROR("OBJDET_CONFIG is not specified");
      std::runtime_error("environment variable OBJDET_CONFIG is not specified");
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

    this->maxModelLimit = std::max(config["max_model_limit"].asInt(), 1);

    std::string modelPath = config["model_abs_path"].asString();

    GST_INFO("check model file");
    if (fs::exists(modelPath) == false) {
      GST_ERROR("object detection model not found: %s", modelPath.c_str());
      throw std::runtime_error(std::string("object detection model not found: ") + modelPath);
    }

    int deviceID = std::max(config["device_id"].asInt(), 0);

    // init models
    GST_INFO("Start init %d models", this->maxModelLimit);
    for (int i = 0; i < this->maxModelLimit; i++) {
      GST_INFO("Init %d/%d model", i + 1, this->maxModelLimit);
      Yolov7trt *md;
      try {
        md = new Yolov7trt(modelPath, deviceID, std::to_string(i));
        GST_INFO("Finish init %d/%d model", i + 1, this->maxModelLimit);
      } catch (const std::exception &e) {
        GST_ERROR("Error init %d/%d model: %s", i + 1, this->maxModelLimit, e.what());
        continue;
      }

      this->models.push_back(md);
      uintptr_t modelAddress = reinterpret_cast<uintptr_t>(md);
      this->isUsed[modelAddress] = false;
      GST_INFO("Added %d/%d model", i + 1, this->maxModelLimit);
    }
  };

  int getAvailableCount() {
    std::lock_guard<std::recursive_mutex> lockNow(lock);

    int count = 0;
    for (auto const &[address, used] : this->isUsed) {
      count += used ? 1 : 0;
    }
    GST_DEBUG("available model is %d", count);
    return count;
  };

  bool isAvailable() {
    std::lock_guard<std::recursive_mutex> lockNow(lock);

    for (auto const &[address, used] : this->isUsed) {
      if (used == false) {
        GST_DEBUG("is available=true");
        return true;
      }
    }
    GST_DEBUG("is available=false");
    return false;
  };

  Yolov7trt *getModel() {
    std::lock_guard<std::recursive_mutex> lockNow(lock);
    GST_INFO("get a free model");
    for (Yolov7trt *model : this->models) {
      uintptr_t address = reinterpret_cast<uintptr_t>(model);
      if (this->isUsed[address] == false) {
        this->isUsed[address] = true;
        GST_INFO("get a model successfully");
        return model;
      }
    }

    GST_WARNING("try to get a model but no model is available");
    return nullptr;
  }

  void returnModel(Yolov7trt *model) {
    GST_INFO("return a model");
    std::lock_guard<std::recursive_mutex> lockNow(lock);
    uintptr_t address = reinterpret_cast<uintptr_t>(model);
    this->isUsed[address] = false;
  }

  ~ModelPool() {
    GST_INFO("destroy models");
    for (Yolov7trt *m : models) {
      delete m;
    }
  };

private:
  std::vector<Yolov7trt *> models;
  std::map<uintptr_t, bool> isUsed;
  int maxModelLimit;
  std::recursive_mutex lock;
};

} // namespace objdet
} // namespace module
} // namespace kurento