//
//  LossFn.hpp
//  dendrite
//
//  Created by Oli Callaghan on 27/11/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef LossFn_hpp
#define LossFn_hpp

#include <stdio.h>
#include <functional>
#include "Loss.hpp"
#include "Tensor.hpp"
#include "Exceptions.hpp"

namespace Loss {
    // Loss function structure
    struct LossFn {
        Loss::Loss_T loss_t;
        //std::function<float(Tensor*, Tensor*, dispatch_queue_t*)> Loss_Val;
        float (*Loss_Val)(Tensor*, Tensor*, dispatch_queue_t*);
        //std::function<void(Tensor*, Tensor*, Tensor*, dispatch_queue_t*)> Loss;
        void (*Loss)(Tensor*, Tensor*, Tensor*, dispatch_queue_t*);
        LossFn(Loss::Loss_T);
    };
}

#endif /* LossFn_hpp */
