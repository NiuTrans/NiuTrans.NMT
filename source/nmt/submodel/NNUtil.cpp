/* NiuTrans.NMT - an open-source neural machine translation system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2020-03-21
 */

#include "NNUtil.h"

/* the nmt namespace */
namespace nmt
{

/* 
a wrapper for the gather function 
>> src - the input tensor
>> index - the index tensor
<< res - the output tensor
*/
XTensor AutoGather(XTensor& src, XTensor& index)
{

    if (src.order == 2) {
        return Gather(src, index);
    } else if (src.order == 3) {
        CheckNTErrors(src.order == 3, "the source must be 3d");

        int order = src.order;
        int dimSize[MAX_TENSOR_DIM_NUM];
        for (int i = 0; i < src.order; i++) {
            dimSize[i] = src.dimSize[i];
        }

        src.Reshape(src.dimSize[0], src.dimSize[1] * src.dimSize[2]);
        XTensor res = Gather(src, index);

        src.Reshape(order, dimSize);

        dimSize[0] = index.dimSize[0];
        dimSize[1] = res.unitNum / (dimSize[0] * dimSize[2]);

        res.Reshape(order, dimSize);
        return res;
    } else {
#ifdef USE_CUDA
        // This branch assumes that src has the shape with (N, B, L, H).
        CheckNTErrors(src.order == 4, "The order of the input tensor must be 4!");
        CheckNTErrors(index.order == 1, "The order of the index tensor must be 1!");
    
        int order = src.order;
        int* dimSize = new int[order];
      
        for(int i=0;i<src.order;i++) {
          if(i==1) dimSize[i] = index.unitNum;
          else dimSize[i] = src.dimSize[i];
        }
    
        float dr = (!src.isSparse) ? 1.0F : src.denseRatio;
        XTensor t(order, dimSize, src.dataType, dr, src.devID, src.mem);
        t.SetTMPFlag();
        const struct UpdateStateParams params{dimSize[0], src.dimSize[1], dimSize[1], dimSize[2], dimSize[3]};
        updateState(&src, &index, params, &t);
        delete [] dimSize;
        return t;
#else
        ShowNTErrors("the source must be 2d or 3d!");
#endif
    }
}

} /* end of the nmt namespace */