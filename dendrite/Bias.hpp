//
//  Bias.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Bias_hpp
#define Bias_hpp

#include <stdio.h>
#include <OpenCL/opencl.h>
#include "LearnableParameters.hpp"

namespace Layers {
    namespace Bias {
        void Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void BackpropDeltas(Tensor** deltas, Tensor* backpropagated_deltas, Tensor* learnable_params, void* params, dispatch_queue_t* queue);
        void CalcParamDeltas(Tensor** deltas, Tensor* kernel_deltas, Tensor* input, void* params, dispatch_queue_t* queue);
        Dims CalcOutputSize(Dims input);
    }
}

#endif /* Bias_hpp */
