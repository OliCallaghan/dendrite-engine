//
//  Layers.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Layers_hpp
#define Layers_hpp

#include <stdio.h>

// Import Layers
#include "BatchNormalisation.hpp"
#include "Bias.hpp"
#include "Convolution.hpp"
#include "Dropout.hpp"
#include "FullyConnected.hpp"
#include "Operation.hpp"
#include "Pooling.hpp"

namespace Layers {
    enum Layer_T {BatchNormalisation_T, Bias_T, Convolution_T, Dropout_T, FullyConnected_T, Input_T, Operation_T, Pooling_T};
}

#endif /* Layers_hpp */
