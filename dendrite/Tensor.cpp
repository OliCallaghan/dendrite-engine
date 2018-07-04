//
//  Tensor.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <functional>
#include "Tensor.hpp"

// Dimensions construction (from array)
Dims::Dims(std::vector<int> d) {
    // Removes invalid dimensions.
    this->dims = new int[4];
    
    if (d.size() > 4) {
        throw TensorDimsErr("too many dimensions, (must be less than 4!)");
    }
    for (unsigned int it = 0; it < d.size(); it++) {
        if (d.at(it) <= 0) {
            std::cerr << "Invalid dimension"; this->dims[it] = 1;
            throw TensorDimsErr("all dimensions must be positive");
        } else {
            this->dims[it] = d.at(it);
        }
    }
}

// Outputs size of dimensions
int Dims::Size() const {
    return this->dims[0] * this->dims[1] * this->dims[2] * this->dims[3];
}

int Dims::SizePerEx() {
    return this->dims[0] * this->dims[1] * this->dims[2];
}

// Returns size of dimensions in user friendly view
std::string Dims::GetSizeStr() {
    std::string str;
    for (int dim = 0; dim < 4; dim++) {
        str.append(std::to_string(this->dims[dim]));
        str.append(" ");
    }
    return str;
}

// Returns size of dimensions in user friendly view
std::string Dims::GetSizeStr(std::string delimiter) {
    std::string str;
    for (int dim = 0; dim < 4; dim++) {
        str.append(std::to_string(this->dims[dim]));
        str.append(delimiter);
    }
    str.pop_back();
    return str;
}

// Initialise tensor with given dimensions
Tensor::Tensor(Dims d) : dims(d) {
    // size = product of all 4 dimensions
    if (d.Size() <= 0) {
        throw TensorDimsErr(d.GetSizeStr());
    }
    
    size_t size = this->dims.Size() * sizeof(float);
    
    this->data = (float*)malloc(size);
}

// Initialise tensor with array dimenisons
Tensor::Tensor(std::vector<int> d) : dims(d) {
    // size = product of all 4 dimensions
    size_t size = this->dims.Size() * sizeof(float);
    
    this->data = (float*)malloc(size);
}

// Returns tensor data as string in user friendly manner (used for debugging)
std::string Tensor::GetDataStr() const {
    std::string str;
    str.append("[ ");
    for (int pos = 0; pos < this->dims.Size(); pos++) {
        str.append(std::to_string(this->data[pos]));
        str.append(" ");
    }
    str.append("]");
    return str;
}

// Returns tensor data for MNIST data (used for debugging)
// Much easier to understand in the console and interpret, rather than list of numbers, so used during development
std::string Tensor::GetMNISTDataStr() {
    std::string str;
    for (int pos = 0; pos < this->dims.Size(); pos++) {
        if (this->data[pos] < 0) {
            str.append(" ");
        } else {
            str.append("X");
        }
        if (pos % 28 == 0) {
            str.append("\n");
        }
    }
    return str;
}
