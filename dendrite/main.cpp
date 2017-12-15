//
//  main.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include "Tensor.hpp"
#include "Network.hpp"

int main(int argc, const char * argv[]) {
    Tensor input({4,1,1,1});
    Tensor prediction({3,1,1,1});
    Tensor output({3,1,1,1});
    
    // Set Input Data
    input.data[0] = 0.8;
    input.data[1] = -0.4;
    input.data[2] = 1;
    input.data[3] = 0.1;
    
    // Set Prediction Data
    prediction.data[0] = 1;
    prediction.data[1] = 0;
    prediction.data[2] = 1;
    
    Network n(&input, &prediction, &output);
    n.LearningRate = 1e-11;
    
    /*float loss;
    for (int loop = 0; loop < 1e10; loop++) {
        loss = n.Learn();
        std::cout << "LOSS: " << loss << "\n";
    }*/
    
    n.ImportNetwork();
    
    return 0;
}
