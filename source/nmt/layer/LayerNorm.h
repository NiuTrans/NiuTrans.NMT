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

#ifndef __LAYERNORMAL_H__
#define __LAYERNORMAL_H__

#include "../Utility.h"
#include "../../niutensor/network//XNet.h"

using namespace nts;

namespace nmt
{

/* layer normalization: y = norm(x) * w + b
   where norm(x) = (x - mean)/standardDeviation */
class LN
{
public:
    /* device id */
    int devID;

    /* the transformation matrix w */
    XTensor weight;

    /* the bias term b */
    XTensor bias;

    /* dimension size of the model */
    int d;

public:
    /* constructor */
    LN();

    /* de-constructor */
    ~LN();

    /* initialize the model */
    void InitModel(Config& config);

    /* make the network */
    XTensor Make(XTensor& input);
};

}

#endif
