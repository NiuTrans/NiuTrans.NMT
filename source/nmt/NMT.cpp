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
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-06, 2020-07
 */

#include <ctime>

#include "NMT.h"
#include "train/Trainer.h"
#include "translate/Translator.h"

namespace nmt
{

int NMTMain(int argc, const char** argv)
{
    if (argc == 0)
        return 1;

    /* load configurations */
    Config config(argc, argv);

    srand(1);

    /* training */
    if (strcmp(config.trainFN, "") != 0) {
        
        Model model;
        model.InitModel(config);

        TensorList params;
        model.GetParams(params);
        int count = 0;
        for (int i = 0; i < params.count; i++)
            count += params[i]->unitNum;
        LOG("number of parameters: %d", count);

        Trainer trainer;
        trainer.Init(config);
        trainer.Train(config.trainFN, config.validFN, config.modelFN, &model);
    }

    /* translating */
    if (strcmp(config.testFN, "") != 0 && strcmp(config.outputFN, "") != 0) {
        
        /* disable grad flow */
        DISABLE_GRAD;

        Model model;
        model.InitModel(config);
        Translator translator;
        translator.Init(config);
        translator.Translate(config.testFN, config.srcVocabFN, 
                             config.tgtVocabFN, config.outputFN, &model);
    }

    return 0;
}

}