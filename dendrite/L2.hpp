//
//  L2.hpp
//  dendrite
//
//  Created by Oli Callaghan on 23/11/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef L2_hpp
#define L2_hpp

#include <stdio.h>
#include <OpenCL/opencl.h>
#include "Tensor.hpp"
#include "L2.cl.h"

// L2 loss layer methods
namespace Loss {
    namespace L2 {
        float LossVal(Tensor* output, Tensor* prediction, dispatch_queue_t* queue);
        void Loss(Tensor* output, Tensor* prediction, Tensor* deltamap, dispatch_queue_t* queue);
    }
}

#endif /* L2_hpp */
