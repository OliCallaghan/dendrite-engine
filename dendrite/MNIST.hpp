//
//  MNIST.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/12/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef MNIST_hpp
#define MNIST_hpp

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include "Tensor.hpp"

// MNIST handler (used in development prior to implementations of data pipelines)
class MNISTHandler {
    long pos_i;
    long pos_l;
    std::string location;
    std::ifstream f_img; // Images
    std::ifstream f_lab; // Labels
    
public:
    int Classify(Tensor*);
    bool VerifyDataSet();
    void LoadData(Tensor*, Tensor*);
    MNISTHandler(std::string loc);
    ~MNISTHandler();
};

#endif /* MNIST_hpp */
