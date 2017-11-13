//
//  Convolution.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Convolution_hpp
#define Convolution_hpp

#include <stdio.h>
#include "LearnableParameters.hpp"

namespace Layers {
    namespace Convolution {
        enum Padding_T {
            Valid, Same
        };
        
        struct Hyperparameters {
            Dims KernelSize;
            Dims Stride; // Only 2
            Padding_T Padding;
            Dims Dilation; // Only 2
        };
        
        Dims CalcOutputSize(Dims input, Hyperparameters params);
    }
}

#endif /* Convolution_hpp */
