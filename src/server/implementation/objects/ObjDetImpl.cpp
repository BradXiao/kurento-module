/* Autogenerated with kurento-module-creator */

#include "ObjDetImpl.hpp"
#include "MediaPipeline.hpp"
#include "MediaPipelineImpl.hpp"
#include <KurentoException.hpp>
#include <ObjDetImplFactory.hpp>
#include <gst/gst.h>
#include <jsonrpc/JsonSerializer.hpp>

GST_DEBUG_CATEGORY_STATIC(kurento_obj_det);
#define GST_CAT_DEFAULT kurento_obj_det

namespace kurento {
namespace module {
namespace objdet {

ObjDetImpl::ObjDetImpl(const boost::property_tree::ptree &config, std::shared_ptr<MediaPipeline> mediaPipeline)
    : OpenCVFilterImpl(config, std::dynamic_pointer_cast<MediaPipelineImpl>(mediaPipeline)) {}

MediaObjectImpl *ObjDetImplFactory::createObject(const boost::property_tree::ptree &config,
                                                 std::shared_ptr<MediaPipeline> mediaPipeline) const {
  return new ObjDetImpl(config, mediaPipeline);
}

ObjDetImpl::StaticConstructor ObjDetImpl::staticConstructor;

ObjDetImpl::StaticConstructor::StaticConstructor() {
  GST_DEBUG_CATEGORY_INIT(kurento_obj_det, "KurentoObjDetImpl", 0, "KurentoObjDetImpl debug category");
}

void ObjDetImpl::setConfidence(float confidence) {
  GST_INFO("set confidence %f", confidence);
  ObjDetOpenCVImpl::setConfidence(confidence);
};
void ObjDetImpl::setBoxLimit(int boxLimit) {
  GST_INFO("set box limit %d", boxLimit);
  ObjDetOpenCVImpl::setBoxLimit(boxLimit);
};
void ObjDetImpl::setIsDraw(bool isDraw) {
  GST_INFO("set is draw %s", isDraw ? "true" : "false");
  ObjDetOpenCVImpl::setIsDraw(isDraw);
};
void ObjDetImpl::startInferring() {
  GST_INFO("start inferring");
  ObjDetOpenCVImpl::startInferring();
};
void ObjDetImpl::stopInferring() {
  GST_INFO("stop inferring");
  ObjDetOpenCVImpl::stopInferring();
};
void ObjDetImpl::heartbeat(const std::string &sessionId) {
  GST_INFO("heartbeat %s", sessionId.c_str());
  ObjDetOpenCVImpl::heartbeat(sessionId);
};
void ObjDetImpl::initSession() {
  GST_INFO("check model state");
  ObjDetOpenCVImpl::initSession();
}
void ObjDetImpl::destroy() {
  GST_INFO("destroy");
  ObjDetOpenCVImpl::destroy();
};

} // namespace objdet
} // namespace module
} // namespace kurento
