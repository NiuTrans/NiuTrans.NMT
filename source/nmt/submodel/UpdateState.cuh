#ifndef __UPDATESTATE_CUH_
#define __UPDATESTATE_CUH_
#include <stdint.h>
#include "../../niutensor/tensor/XGlobal.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/XDevice.h"

using namespace nts;
namespace nmt {
  struct UpdateStateParams {
    const uint32_t num_head;
    const uint32_t src_batch_size;
    const uint32_t tgt_batch_size;
    const uint32_t seqlen;
    const uint32_t head_dim;
  };
  
  void updateState(XTensor *s, XTensor* index, struct UpdateStateParams params, XTensor* t);
}
#endif