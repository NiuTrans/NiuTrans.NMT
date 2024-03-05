#ifndef __UPDATESTATE_CUH_
#define __UPDATESTATE_CUH_
#include <stdint.h>
#include "../../niutensor/tensor/XGlobal.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/XDevice.h"

using namespace nts;
namespace nmt {
  struct UpdateStateParams {
    const int num_head;
    const int src_batch_size;
    const int tgt_batch_size;
    const int seqlen;
    const int head_dim;
  };
  
  void updateState(XTensor *s, XTensor* index, struct UpdateStateParams params, XTensor* t);
}
#endif