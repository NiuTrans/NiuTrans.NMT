/* NiuTrans.NMT - an open-source neural machine translation system.
 * Copyright (C) 2024 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __UPDATESTATE_CUH__
#define __UPDATESTATE_CUH__
#include <stdint.h>
#include "../../niutensor/tensor/XGlobal.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/XDevice.h"

using namespace nts;
namespace nmt {

#ifdef USE_CUDA

  struct UpdateStateParams {
    uint32_t num_head;
    uint32_t src_batch_size;
    uint32_t tgt_batch_size;
    uint32_t seqlen;
    uint32_t head_dim;
  };

  /* 
  Update kv cache state.
  >> src - (N, B, L, H)
  >> index - (B)
  << tgt - (N, B, L, H)
  */
  void updateState(const XTensor* const s,
                   const XTensor* const index,
                   const struct UpdateStateParams params,
                   XTensor* const t);
#endif // USE_CUDA
} /* end of the nmt namespace */

#endif // __UPDATESTATE_CUH__