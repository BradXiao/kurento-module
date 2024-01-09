/* Autogenerated with kurento-module-creator */
#include "ObjDetOpenCVImpl.hpp"
#include "ModelPool.hpp"
#include <KurentoException.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <chrono>
#include <gst/gst.h>

GST_DEBUG_CATEGORY_STATIC(kurento_obj_det_core);
#define GST_CAT_DEFAULT kurento_obj_det_core

namespace kurento {
namespace module {
namespace objdet {

static ModelPool modelPool;

ObjDetOpenCVImpl::ObjDetOpenCVImpl() {
  this->sessionId = boost::uuids::to_string(uuid_gen());
  GST_DEBUG_CATEGORY_INIT(kurento_obj_det_core, (std::string("ObjDetCore-") + this->sessionId).c_str(), GST_DEBUG_FG_CYAN,
                          "ObjDetCore");
  GST_INFO("session started %s", this->sessionId.c_str());
}

/*
 * This function will be called with each new frame. mat variable
 * contains the current frame. You should insert your image processing code
 * here. Any changes in mat, will be sent through the Media Pipeline.
 */
void ObjDetOpenCVImpl::process(cv::Mat &mat) {
  GST_DEBUG("process");
  if (this->sessionId == "") {
    GST_WARNING("session is not init");
    sendErrorMessage("E001", "session is not init");
    return;
  }
  if (this->model == nullptr) {
    GST_DEBUG("model is nullptr");
    sendErrorMessage("E002", "model is unavailable");
    return;
  }

  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  if (now - this->sessionCheckTimestamp > 60) {
    if (objdet::modelPool.sessionExists(this->modelName, this->sessionId) == false) {
      GST_WARNING("session expired %s", this->sessionId.c_str());
      sendErrorMessage("E003", "session expired");
      this->model = nullptr;
      return;
    }

    this->sessionCheckTimestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  }

  if (this->isInferring == true) {
    GST_DEBUG("do inferring");
    std::vector<utils::Obj> objs;
    // infer
    GST_DEBUG("feed mat into model");
    this->model->infer(mat, objs);
    GST_DEBUG("inferred %d objs", static_cast<int>(objs.size()));
    std::vector<utils::Obj> objsTmp;
    // confidence
    for (utils::Obj &obj : objs) {
      if (obj.confi >= this->confiThresh) {
        auto it = std::lower_bound(objsTmp.begin(), objsTmp.end(), obj);
        objsTmp.insert(it, obj);
      }
    }
    objs = objsTmp;
    GST_DEBUG("%d objs are above confidence %f", static_cast<int>(objs.size()), this->confiThresh);
    // box limit
    if (static_cast<int>(objs.size()) > this->boxLimit) {
      std::vector<utils::Obj> objsTmp(objs.begin(), objs.begin() + std::min(objs.size(), size_t(this->boxLimit)));
      objs = objsTmp;
      GST_DEBUG("%d objs after truncating additional objs, max= %d", static_cast<int>(objs.size()), this->boxLimit);
    }

    // draw box
    if (this->isDraw == true && objs.size() > 0) {
      GST_DEBUG("draw objs");
      utils::drawObjsFixedColor(mat, mat, objs, false, 0.4, utils::CLASSCOLORS);
    }

    Json::Value boxes(Json::arrayValue);
    for (utils::Obj &obj : objs) {
      Json::Value box;
      box["x1"] = obj.p1.x;
      box["y1"] = obj.p1.y;
      box["x2"] = obj.p2.x;
      box["y2"] = obj.p2.y;
      box["name"] = obj.name;
      box["confi"] = obj.confi;
      boxes.append(box);
    }
    GST_DEBUG("signalboxDetected");
    boxDetected event(this->getSharedFromThis(), boxDetected::getName(), utils::jsonToString(boxes));
    signalboxDetected(event);
  } else {
    GST_DEBUG("no inferring");
  }
}

bool ObjDetOpenCVImpl::setConfidence(float confidence) {
  GST_INFO("set confidence to %f", confidence);
  if (confidence <= 0 || confidence > 1) {
    GST_WARNING("confidence set error");
    this->sendSetParamSetResult("confidence", "E001");
    return false;
  }
  this->confiThresh = std::min(std::max(confidence, 0.01f), 0.99f);
  this->sendSetParamSetResult("confidence", "000");
  return true;
}

bool ObjDetOpenCVImpl::setBoxLimit(int boxLimit) {
  GST_INFO("set boxLimit to %d", boxLimit);
  if (boxLimit <= 0 || boxLimit > 100) {
    GST_WARNING("boxLimit set error");
    this->sendSetParamSetResult("boxLimit", "E001");
    return false;
  }
  this->boxLimit = std::min(std::max(boxLimit, 1), 100);
  this->sendSetParamSetResult("boxLimit", "000");
  return true;
}

bool ObjDetOpenCVImpl::setIsDraw(bool isDraw) {
  GST_INFO("set isDraw to %s", isDraw ? "true" : "false");
  this->isDraw = isDraw;
  this->sendSetParamSetResult("isDraw", "000");
  return true;
}

bool ObjDetOpenCVImpl::startInferring() {
  GST_INFO("set isInferring to true");
  this->isInferring = true;
  this->sendSetParamSetResult("startinferring", "000");
  return true;
}
bool ObjDetOpenCVImpl::stopInferring() {
  GST_INFO("set isInferring to false");
  this->isInferring = false;
  this->sendSetParamSetResult("stopinferring", "000");
  return true;
}
bool ObjDetOpenCVImpl::heartbeat(std::string sessionId) {
  GST_DEBUG("heartbeat %s", sessionId.c_str());
  objdet::modelPool.heartbeat(this->modelName, sessionId);
  return true;
}
bool ObjDetOpenCVImpl::initSession() {
  this->initSession("default");
  return true;
}

bool ObjDetOpenCVImpl::changeModel(const std::string &modelName) {
  GST_INFO("change model");
  bool isInfer = this->isInferring;
  this->isInferring = false;

  if (this->model != nullptr) {
    objdet::modelPool.returnModel(this->modelName, this->model, this->sessionId);
    this->model = nullptr;
  }
  this->initSession(modelName);

  this->isInferring = isInfer; // restore state
  return true;
}

bool ObjDetOpenCVImpl::getModelNames() {
  GST_INFO("get model names");
  Json::Value modelNamesJson(Json::arrayValue);
  std::vector<std::string> modelNames;
  objdet::modelPool.getModelNames(modelNames);

  for (const auto &name : modelNames) {
    modelNamesJson.append(name);
  }

  modelNamesEvent event(this->getSharedFromThis(), modelNamesEvent::getName(), utils::jsonToString(modelNamesJson));
  signalmodelNamesEvent(event);
  return true;
}

bool ObjDetOpenCVImpl::destroy() {
  if (this->model != nullptr) {
    objdet::modelPool.returnModel(this->modelName, this->model, this->sessionId);
    GST_INFO("release a model");
    this->model = nullptr;
    this->sendSetParamSetResult("destroy", "000");
    return true;
  } else {
    GST_WARNING("no model needs to be released");
    this->sendSetParamSetResult("destroy", "W001");
    return false;
  }
}

ObjDetOpenCVImpl::~ObjDetOpenCVImpl() {
  if (this->model != nullptr) {
    objdet::modelPool.returnModel(this->modelName, this->model, this->sessionId);
    GST_INFO("release a model");
  }
}
// ================================================================================================================
// private
// ================================================================================================================

void ObjDetOpenCVImpl::sendSetParamSetResult(const std::string param_name, const std::string state) {
  Json::Value result;
  result["state"] = state;
  result["param_name"] = param_name;
  paramSetState event(this->getSharedFromThis(), paramSetState::getName(), utils::jsonToString(result));
  GST_DEBUG("signalparamSetState");
  signalparamSetState(event);
};

void ObjDetOpenCVImpl::sendErrorMessage(const std::string &state, const std::string &msg) {

  Json::Value result;
  result["state"] = state;
  result["msg"] = msg;
  errorMessage event(this->getSharedFromThis(), errorMessage::getName(), utils::jsonToString(result));
  GST_WARNING("send error message %s,%s", state.c_str(), msg.c_str());
  signalerrorMessage(event);
};

bool ObjDetOpenCVImpl::initSession(const std::string &modelName) {
  GST_INFO("init session");
  this->modelName = modelName;
  if (this->modelName == "default") {
    this->modelName = objdet::modelPool.getDefaultModelName();
  }
  this->model = objdet::modelPool.getModel(this->modelName);

  Json::Value modelState;
  if (this->model != nullptr) {
    this->sessionId = objdet::modelPool.registerSession(this->modelName, this->model, this->sessionId);
    GST_INFO("model is ready");
    modelState["state"] = "000";
    modelState["defaultModel"] = this->modelName;
    modelState["msg"] = "";
    modelState["sessionId"] = this->sessionId;
    this->sessionCheckTimestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  } else {
    GST_WARNING("no model is available");
    modelState["state"] = "E005";
    modelState["defaultModel"] = "";
    modelState["msg"] = "Model not avaiable or not found";
    modelState["sessionId"] = "";
  }

  sessionInitState event(this->getSharedFromThis(), sessionInitState::getName(), utils::jsonToString(modelState));
  signalsessionInitState(event);
  return true;
};

} // namespace objdet
} // namespace module
} // namespace kurento
