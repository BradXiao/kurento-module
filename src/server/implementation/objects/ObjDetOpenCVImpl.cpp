/* Autogenerated with kurento-module-creator */

#include "ObjDetOpenCVImpl.hpp"
#include <KurentoException.hpp>

namespace kurento {
namespace module {
namespace objdet {

ObjDetOpenCVImpl::ObjDetOpenCVImpl() {}

/*
 * This function will be called with each new frame. mat variable
 * contains the current frame. You should insert your image processing code
 * here. Any changes in mat, will be sent through the Media Pipeline.
 */
void ObjDetOpenCVImpl::process(cv::Mat &mat) {
  // FIXME: Implement this
  throw KurentoException(NOT_IMPLEMENTED, "ObjDetOpenCVImpl::process: Not implemented");
}

} // namespace objdet
} // namespace module
} // namespace kurento
