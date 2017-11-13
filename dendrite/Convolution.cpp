//
//  Convolution.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "Convolution.hpp"
#include <math.h>

Dims Layers::Convolution::CalcOutputSize(Dims input, Hyperparameters Parameters) {
    Dims output = input;
    if (Parameters.Padding == Layers::Convolution::Padding_T::Valid) {
        // Valid Convolution
        output.dims[0] = ceil(float(input.dims[0] - Parameters.KernelSize.dims[0] + 1) / float(Parameters.Stride.dims[0]));
        output.dims[1] = ceil(float(input.dims[1] - Parameters.KernelSize.dims[1] + 1) / float(Parameters.Stride.dims[1]));
        output.dims[2] = Parameters.KernelSize.dims[3]; // Number of kernels
    } else {
        // Same Convolution
        output.dims[0] = ceil(float(input.dims[0]) / float(Parameters.Stride.dims[0]));
        output.dims[1] = ceil(float(input.dims[1]) / float(Parameters.Stride.dims[1]));
        output.dims[2] = Parameters.KernelSize.dims[3]; // Number of kernels
    }
    
    return output;
}
