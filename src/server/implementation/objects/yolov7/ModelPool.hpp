#include "yolov7.hpp"

#include "utils.hpp"
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gst/gst.h>
#include <json/json.h>
#include <mutex>

namespace fs = std::filesystem;
namespace kurento {
namespace module {
namespace objdet {

class ModelBundle {
public:
  std::vector<Yolov7trt *> models;
  std::map<uintptr_t, bool> isUsed;
  std::map<std::string, std::time_t> sessionHeartbeat;
  std::map<std::string, Yolov7trt *> sessionToModel;

  ~ModelBundle();
};

class ModelPool {
public:
  ModelPool();

  int getAvailableCount(const std::string &modelName);
  bool isAvailable(const std::string &modelName);
  Yolov7trt *getModel(const std::string &modelName);
  std::string getDefaultModelName();
  void returnModel(const std::string &modelName, Yolov7trt *model, const std::string &sessionId);
  void getModelNames(std::vector<std::string> &names);
  bool modelExists(const std::string &modelName);
  void registerSession(const std::string &modelName, Yolov7trt *model, const std::string &sessionId);
  void heartbeat(const std::string &modelName, std::string sessionId);
  bool sessionExists(const std::string &modelName, std::string sessionId);

  ~ModelPool();

private:
  std::map<std::string, ModelBundle *> modelBundles;
  std::recursive_mutex lock;
  std::string defaultModelName;

  void readConfig(Json::Value &config);
  void initModels(const Json::Value &config);
  bool updateSession(const std::string &modelName);
  void checkVRAM(const int deviceId, const size_t minBytes);
};

} // namespace objdet
} // namespace module
} // namespace kurento