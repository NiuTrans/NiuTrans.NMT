#include "UpdateState.cuh"

namespace nmt {

  // src (N, B, L, H)
  // tgt (N, B, L, H)
  // index (B)
  __global__ void updateStateKernel(const float* const src,
                                    const int* const index,
                                    const UpdateStateParams params,
                                    float * const tgt) {
    // niutensor doesn't guarantee 128-bit aligned.
    // TODO(umiswing): Use 128-bit load after 128-bit aligned support.
    constexpr int ELEMENTS_PER_THREAD = 1;
    const int THREADS_PER_BLOCK = blockDim.x;
    const int ELEMENTS_PER_BLOCK = ELEMENTS_PER_THREAD * THREADS_PER_BLOCK;
  
    const int src_hid = blockIdx.y;
    const int src_bid = index[blockIdx.x];
    const int src_hbid = (src_hid * params.src_batch_size + src_bid) * params.seqlen * params.head_dim;
  
    const int tgt_hid = blockIdx.y;
    const int tgt_bid = blockIdx.x;
    const int tgt_hbid = (tgt_hid * params.tgt_batch_size + tgt_bid) * params.seqlen * params.head_dim;
    #pragma unroll
    for(int i=threadIdx.x*ELEMENTS_PER_THREAD;i<params.seqlen * params.head_dim / ELEMENTS_PER_THREAD;i+=ELEMENTS_PER_BLOCK) {
      tgt[tgt_hbid+i] = src[src_hbid+i];
    }
  }
  
  void updateState(XTensor *s, XTensor* index, struct UpdateStateParams params, XTensor* t) {
    int devID = s->devID;
    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    dim3 blocks(params.tgt_batch_size, params.num_head);

    dim3 threads(params.seqlen*params.head_dim > GDevs.GPUs[devID].GPUMaxThreadNumPerBlock ?
                 GDevs.GPUs[devID].GPUMaxThreadNumPerBlock :
                 params.seqlen*params.head_dim );

    updateStateKernel<<<blocks, threads>>>(static_cast<float*>(s->data),
                                           static_cast<int*>(index->data),
                                           params,
                                           static_cast<float*>(t->data));

    BacktoCudaDev(devID, devIDBackup);
  }
}