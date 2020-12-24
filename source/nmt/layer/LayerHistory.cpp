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
 * $Created by: Bei Li (libei_neu@outlook.com) 2020-02-03
 * $Modified by: Chi Hu (huchinlp@gmail.com) 2020-12-10
 */

#include "Embedding.h"
#include "LayerNorm.h"
#include "LayerHistory.h"
#include "../Utility.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/XName.h"

#define SAFE_DELETE(x) do{ if((x) != NULL){delete (x); (x) = NULL;} } while(false)
#define SAFE_DELETE_ARRAY(x) do{ if((x) != NULL) {delete [] (x); (x)=NULL;} } while(false)

namespace nmt
{

/* constructor */
LayerHistory::LayerHistory()
{
    d = -1;
    devID = -1;
    count = -1;
    nlayer = -1;
    weights = NULL;
    history = NULL;
    layerNorms = NULL;
}

/* de-constructor */
LayerHistory::~LayerHistory()
{
    delete history;
    delete[] layerNorms;
    delete[] weights;
}

/*
initialize the model
>> config - configurations of the model
*/
void LayerHistory::InitModel(Config& config)
{
    devID = config.devID;
    d = config.modelSize;
    nlayer = config.nEncLayer;

    /*  the triangle weight matrices for dlcl 
        layer 0: [1, 0, ..., 0]               
        layer 1: [0.5, 0.5, ..., 0]           
        layer 2: [0.33, 0.33, 0.33, ..., 0]   */
    weights = new XTensor[nlayer + 1];
    for (int i = 0; i < nlayer + 1; i++) {
        InitTensor1D(&(weights[i]), i + 1, X_FLOAT, devID);
        float* data = new float[i + 1];
        for (int j = 0; j < i + 1; j++) {
            data[j] = 1.0F / float(i + 1);
        }
        weights[i].SetData(data, i + 1);
        delete[] data;
    }

    layerNorms = new LN[nlayer];

    /* initialize the layer normalization of each layer */
    for (int i = 0; i < nlayer; i++) {
        layerNorms[i].InitModel(config);
    }
}

/*
the Add operation
>> layer - the previous layer output. It might be of size B * L * H
           where B = batch size, L = sequence length,
           and H = vector size of each position
*/
void LayerHistory::Add(XTensor& layer)
{
    /* the embedding is not normed */
    count += 1;
    if (history->count == 0) {
        history->Add(layer);
        return;
    }
    layer = layerNorms[count - 2].Make(layer);
    history->Add(layer);
}

/*
calculate the weighted sum of previous layers
the result for the i-th layer is:
result = sum(layers[0...i] * weight[i][0...i])
shape of the result: B * L * H
*/
XTensor LayerHistory::Pop()
{
    TensorList list;
    for (int i = 0; i < history->count; i++) {
        list.Add(&(history->list[i]));
    }
    XTensor stack;
    stack = Merge(list, 0);
    //Stack(list, 0);

    int dimSize[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < stack.order + 1; i++)
        dimSize[i + 1] = stack.dimSize[i];
    dimSize[0] = list.Size();
    dimSize[1] /= dimSize[0];
    stack = Reshape(stack, stack.order + 1, dimSize);

    XTensor res;
    res = MultiplyDim(stack, weights[list.Size() - 1], 0);

    return ReduceSum(res, 0);
}

/* clear the history */
void LayerHistory::ClearHistory(bool reset)
{
    if(history != NULL)
        delete history;
    if(reset)
        history = new History;
    else
        history = NULL;
    count = 0;
}

/* initialize the history */
History::History()
{
    count = 0;
}

/* delete the history */
History::~History()
{
    for (int i = 0; i < MAX_LAYER_NUM; i++) {
        list[i].DestroyData();
        XLink::ClearOutgoing(&list[i]);
        XLink::ClearIncoming(&list[i]);
        if (list[i].grad != NULL)
            delete list[i].grad;
    }
}

/* append a layer to the history */
void History::Add(XTensor& layer)
{
    list[count] = std::move(layer);
    XLink::ClearOutgoing(&layer);
    XLink::ClearIncoming(&layer);
    count++;
}

}