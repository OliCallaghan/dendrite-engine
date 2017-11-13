//
//  Pooling.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Pooling_hpp
#define Pooling_hpp

#include <stdio.h>
#include "Tensor.hpp"

namespace Layers {
    namespace Pooling {
        enum Pooling_T {
            Max, Average
        };
        
        struct Hyperparameters {
            Pooling_T Method;
            Dims PoolSize;
            Dims Stride;
        };
    }
}

#endif /* Pooling_hpp */
