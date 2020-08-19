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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "FNN.h"
#include "Embedding.h"
#include "../Utility.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/function/FHeader.h"

namespace nmt
{

/* constructor */
FNN::FNN()
{
    inSize = -1;
    outSize = -1;
    hSize = -1;
}

/* de-constructor */
FNN::~FNN()
{
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> config - configurations of the model
*/
void FNN::InitModel(Config& config)
{
    devID = config.devID;

    inSize = config.modelSize;
    outSize = config.modelSize;
    hSize = config.fnnHiddenSize;
    dropoutP = config.fnnDropout;

    InitTensor2D(&w1, inSize, hSize, X_FLOAT, devID);
    InitTensor1D(&b1, hSize, X_FLOAT, devID);

    InitTensor2D(&w2, hSize, outSize, X_FLOAT, devID);
    InitTensor1D(&b2, outSize, X_FLOAT, devID);

    float scale = 1.0F;
    _SetDataFanInOut(&w1, scale);
    _SetDataFanInOut(&w2, scale);

    w1.SetDataRand(-(DTYPE)sqrt(6.0F / inSize), (DTYPE)sqrt(6.0F / inSize));
    w2.SetDataRand(-(DTYPE)sqrt(6.0F / hSize), (DTYPE)sqrt(6.0F / hSize));

    b1.SetZeroAll();
    b2.SetZeroAll();
}

/*
make the network
y = max(0, x * w1 + b1) * w2 + b2
>> input - the input tensor
>> return - the output tensor
*/
XTensor FNN::Make(XTensor& input, bool isTraining)
{
    XTensor t1;

    /* t1 = max(0, x * w1 + b1) */
    t1 = Rectify(MulAndShift(input, w1, b1));

    if (isTraining && dropoutP > 0)
        t1 = Dropout(t1, dropoutP);

    /* result = t1 * w2 + b2 */
    return MulAndShift(t1, w2, b2);
}

}