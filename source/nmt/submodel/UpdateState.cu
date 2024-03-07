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

#include "UpdateState.cuh"

namespace nmt {

#ifdef USE_CUDA

  __global__ void updateStateKernel(const float* const src,
                                    const int* const index,
                                    const UpdateStateParams params,
                                    const int split_size,
                                    float * const tgt) {
    // TODO: niutensor doesn't guarantee 128-bit aligned.
    // Use 128-bit loading after 128-bit aligned support.
    constexpr int elements_per_thread = 1;
    const int threads_per_block = blockDim.x;
    const int elements_per_block = elements_per_thread * threads_per_block;
    // split_size for current tb
    const int current_split_size = blockIdx.x == gridDim.x-1 ?
                                   params.seqlen * params.head_dim - blockIdx.x * split_size :
                                   split_size;
  
    const int src_hid = blockIdx.z;
    const int src_bid = index[blockIdx.y];
    const int src_split_id = blockIdx.x;
    const int src_hb_split_id = (src_hid * params.src_batch_size + src_bid) * params.seqlen * params.head_dim + src_split_id * split_size;
  
    const int tgt_hid = blockIdx.z;
    const int tgt_bid = blockIdx.y;
    const int tgt_split_id = blockIdx.x;
    const int tgt_hb_split_id = (tgt_hid * params.tgt_batch_size + tgt_bid) * params.seqlen * params.head_dim + tgt_split_id * split_size;
    #pragma unroll
    for(int i=threadIdx.x*elements_per_thread;i<current_split_size;i+=elements_per_block) {
      tgt[tgt_hb_split_id+i] = src[src_hb_split_id+i];
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

    // shorthand for seqlen*head_dim
    const auto sh_num = params.seqlen * params.head_dim;

    // split along (L, H)
    // TODO: add heuristic (maybe)
    const int loads_per_block = UPDATE_STATE_LOADS_PER_BLOCK;
    const int max_split_size = GDevs.GPUs[devID].GPUMaxThreadNumPerBlock * loads_per_block;
    const int split_size = sh_num > max_split_size ? max_split_size : sh_num;
    const int num_split = (sh_num + split_size - 1) / split_size;

    dim3 blocks(num_split, params.tgt_batch_size, params.num_head);

    dim3 threads((split_size + loads_per_block - 1) / loads_per_block);

    updateStateKernel<<<blocks, threads>>>(static_cast<float*>(src->data),
                                           static_cast<int*>(index->data),
                                           params,
                                           split_size,
                                           static_cast<float*>(tgt->data));

    BacktoCudaDev(devID, devIDBackup);
  }
#endif // USE_CUDA

} /* end of the nmt namespace */