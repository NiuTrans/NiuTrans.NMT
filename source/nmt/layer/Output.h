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

#ifndef __OUTPUT_H__
#define __OUTPUT_H__

#include "../Utility.h"
#include "../../niutensor/tensor/function/FHeader.h"

using namespace nts;

namespace nmt
{

/* output layer */
class Output
{
public:
    /* device id */
    int devID;

    /* vocabulary size */
    int vSize;

    /* vector size of the linear transformation */
    int hSize;

    /* the padding index */
    int padIdx;

    /* transformation matrix */
    XTensor w;

public:
    /* constructor */
    Output();

    /* de-constructor */
    ~Output();

    /* initialize the model */
    void InitModel(Config& config);

    /* make the network (redefined output tensor) */
    void Make(XTensor& input, XTensor& output, bool isTraining, bool normalized);
};

}

#endif