/* Autogenerated with kurento-module-creator */

#ifndef __OBJ_DET_OPENCV_IMPL_HPP__
#define __OBJ_DET_OPENCV_IMPL_HPP__
#include "yolov7.hpp"

#include "ObjDet.hpp"
#include <EventHandler.hpp>
#include <OpenCVProcess.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace kurento {
namespace module {
namespace objdet {

class ObjDetOpenCVImpl : public virtual OpenCVProcess {

public:
  ObjDetOpenCVImpl();
  ~ObjDetOpenCVImpl();

  virtual void process(cv::Mat &mat);
  virtual std::shared_ptr<MediaObject> getSharedFromThis() = 0;

  sigc::signal<void, boxDetected> signalboxDetected;
  sigc::signal<void, sessionInitState> signalsessionInitState;
  sigc::signal<void, paramSetState> signalparamSetState;

  bool setConfidence(float confidence);
  bool setBoxLimit(int boxLimit);
  bool setIsDraw(bool isDraw);
  bool startInferring();
  bool stopInferring();
  bool heartbeat(std::string sessionId);
  bool initSession();
  bool destroy();

private:
  std::string sessionId;
  float confiThresh = 0.7;
  int boxLimit = 10;
  bool isDraw = false;
  bool isInferring = false;
  Yolov7trt *model;
  std::time_t sessionCheckTimestamp;
  void sendSetParamSetResult(const std::string param_name, const std::string state);
  boost::uuids::random_generator uuid_gen;
};

} // namespace objdet
} // namespace module
} // namespace kurento

#endif /*  __OBJ_DET_OPENCV_IMPL_HPP__ */
