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
#include "Bias.hpp"
#include "FullyConnected.hpp"

// Activations
#include "Logistic.hpp"
#include "LinearUnit.hpp"
#include "Softmax.hpp"
#include "Tangent.hpp"

namespace Layers {
    enum Layer_T {Bias_T, FullyConnected_T, Input_T, ReLU_T, Logistic_T, Softmax_T, Tangent_T};
}

#endif /* Layers_hpp */
