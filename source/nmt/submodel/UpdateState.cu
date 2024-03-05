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

/*
 * $Created by: umiswing (umiswing@foxmail.com) 2024-03
 */

#include "UpdateState.cuh"

namespace nmt {

#ifdef USE_CUDA

  __global__ void updateStateKernel(const float* const src,
                                    const int* const index,
                                    const UpdateStateParams params,
                                    float * const tgt) {
    // TODO(umiswing): niutensor doesn't guarantee 128-bit aligned.
    // Use 128-bit loading after 128-bit aligned support.
    constexpr int elements_per_thread = 1;
    const int threads_per_block = blockDim.x;
    const int elements_per_block = elements_per_thread * threads_per_block;
  
    const int src_hid = blockIdx.y;
    const int src_bid = index[blockIdx.x];
    const int src_hbid = (src_hid * params.src_batch_size + src_bid) * params.seqlen * params.head_dim;
  
    const int tgt_hid = blockIdx.y;
    const int tgt_bid = blockIdx.x;
    const int tgt_hbid = (tgt_hid * params.tgt_batch_size + tgt_bid) * params.seqlen * params.head_dim;
    #pragma unroll
    for(int i=threadIdx.x*elements_per_thread;i<params.seqlen * params.head_dim / elements_per_thread;i+=elements_per_block) {
      tgt[tgt_hbid+i] = src[src_hbid+i];
    }
  }

  void updateState(const XTensor* const src,
                   const XTensor* const index,
                   const struct UpdateStateParams params,
                   XTensor* const tgt) {
    CheckNTErrors(src != nullptr &&
                  index != nullptr &&
                  tgt != nullptr,
                  "Invalid tensor!");
    CheckNTErrors(src->dataType == X_FLOAT, "only support state with type X_FLOAT now!");
    CheckNTErrors(tgt->dataType == X_FLOAT, "only support state with type X_FLOAT now!");
    CheckNTErrors(index->dataType == X_INT, "index must be type X_INT!");
    CheckNTErrors(src->devID >= 0, "the state must be kept on the gpu!");
    CheckNTErrors(src->devID == tgt->devID, "the state must be kept on the same device!");
    CheckNTErrors((src->unitSize == tgt->unitSize), "Unmatched tensors!");

    int devID = src->devID;
    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    dim3 blocks(params.tgt_batch_size, params.num_head);

    dim3 threads(params.seqlen*params.head_dim > GDevs.GPUs[devID].GPUMaxThreadNumPerBlock ?
                 GDevs.GPUs[devID].GPUMaxThreadNumPerBlock :
                 params.seqlen*params.head_dim );

    updateStateKernel<<<blocks, threads>>>(static_cast<float*>(src->data),
                                           static_cast<int*>(index->data),
                                           params,
                                           static_cast<float*>(tgt->data));

    BacktoCudaDev(devID, devIDBackup);
  }
#endif // USE_CUDA

} /* end of the nmt namespace */