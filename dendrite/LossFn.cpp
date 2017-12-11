//
//  LossFn.cpp
//  dendrite
//
//  Created by Oli Callaghan on 27/11/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "LossFn.hpp"

Loss::LossFn::LossFn(Loss::Loss_T t) {
    this->loss_t = t;
    switch (t) {
        case Loss::Loss_T::L2_T:
            this->Loss = Loss::L2::Loss;
            this->Loss_Val = Loss::L2::LossVal;
            break;
            
        default:
            break;
    }
}
