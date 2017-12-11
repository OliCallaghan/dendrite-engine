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

namespace Loss {
    struct LossFn {
        Loss::Loss_T loss_t;
        std::function<float(Tensor*, Tensor*, dispatch_queue_t*)> Loss_Val;
        std::function<void(Tensor*, Tensor*, Tensor*, dispatch_queue_t*)> Loss;
        LossFn(Loss::Loss_T);
    };
}

#endif /* LossFn_hpp */
