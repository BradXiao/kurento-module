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
  /// @brief all models 
  std::vector<Yolov7trt *> models;

  /// @brief the usage state of models 
  std::map<uintptr_t, bool> isUsed;

  /// @brief to recycle a model not used for a period of time
  std::map<std::string, std::time_t> sessionHeartbeat;

  /// @brief session-dedicated model map
  std::map<std::string, Yolov7trt *> sessionToModel;

  ~ModelBundle();
};

class ModelPool {
public:
  ModelPool();
  /// @brief get current unused models
  int getAvailableCount(const std::string &modelName);

  /// @brief test if specific type of model has available instances
  bool isAvailable(const std::string &modelName);

  /// @brief get a model by model name
  Yolov7trt *getModel(const std::string &modelName);

  /// @brief get default model name
  std::string getDefaultModelName();

  /// @brief destroy a session and mark the model as available
  void returnModel(const std::string &modelName, Yolov7trt *model, const std::string &sessionId);

  /// @brief get all model names
  void getModelNames(std::vector<std::string> &names);

  /// @brief test if a model name exists
  bool modelExists(const std::string &modelName);

  /// @brief bind a model to a session and mark the model occupied
  void registerSession(const std::string &modelName, Yolov7trt *model, const std::string &sessionId);

  /// @brief heartbeat for a session
  void heartbeat(const std::string &modelName, std::string sessionId);

  /// @brief test if a session exists
  bool sessionExists(const std::string &modelName, std::string sessionId);

  ~ModelPool();

private:
  /// @brief all models keyed by model name
  std::map<std::string, ModelBundle *> modelBundles;

  std::recursive_mutex lock;
  std::string defaultModelName;

  /// @brief read JSON format config file from environment parameter
  void readConfig(Json::Value &config);

  /// @brief init model
  void initModels(const Json::Value &config);

  /// @brief destroy timeout sessions and mark the models available 
  bool updateSession(const std::string &modelName);

  /// @brief check GPU available memory
  void checkVRAM(const int deviceId, const size_t minBytes);
};

} // namespace objdet
} // namespace module
} // namespace kurento