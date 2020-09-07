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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-10
 */

//#define CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

#include "./nmt/NMT.h"
#include "niutensor/network/XNoder.h"
#include "niutensor/tensor/XTensor.h"
#include "niutensor/tensor/core/movement/Spread.h"

using namespace nmt;
using namespace nts;

void test() {
    XTensor input, node, index;
    InitTensor2D(&input, 32, 4);
    InitTensor2D(&input, 13, 4);
    InitTensor2D(&input, 32, 4);

    XNoder::MakeGrad(&input);

    XTensor* tmp = NewTensorBufV2(&input, input.devID, input.mem);
    _SpreadForGather(tmp, node.grad, &index);

    _SumMe(input.grad, tmp);
    input.grad->Dump(stderr);
}

int main(int argc, const char** argv)
{
    //_CrtSetDbgFlag(_CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) | _CRTDBG_LEAK_CHECK_DF);
    //_CrtSetBreakAlloc(2708);

    NMTMain(argc - 1, argv + 1);

    //test();

    //_CrtDumpMemoryLeaks();
    
    return 0;
}

