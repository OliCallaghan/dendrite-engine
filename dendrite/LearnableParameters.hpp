//
//  LearnableParameters.hpp
//  dendrite
//
//  Created by Oli Callaghan on 29/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef LearnableParameters_hpp
#define LearnableParameters_hpp

#include <stdio.h>
#include "Tensor.hpp"

class LearnableParameters: public Tensor {
public:
    using Tensor::Tensor;
    void InitialiseNormal(float mean, float stddev);
    void Update(Tensor);
};

#endif /* LearnableParameters_hpp */
