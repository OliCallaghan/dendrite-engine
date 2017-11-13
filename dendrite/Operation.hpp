//
//  Operation.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Operation_hpp
#define Operation_hpp

#include <stdio.h>

namespace Layers {
    namespace Operation {
        enum Operation_T {
            Addition, Subtraction
        };
        struct Hyperparameters {
            Operation_T Operator;
        };
    }
}

#endif /* Operation_hpp */
