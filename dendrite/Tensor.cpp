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

int Dims::Size() const {
    return this->dims[0] * this->dims[1] * this->dims[2] * this->dims[3];
}

int Dims::SizePerEx() {
    return this->dims[0] * this->dims[1] * this->dims[2];
}

std::string Dims::GetSizeStr() {
    std::string str;
    for (int dim = 0; dim < 4; dim++) {
        str.append(std::to_string(this->dims[dim]));
        str.append(" ");
    }
    return str;
}

std::string Dims::GetSizeStr(std::string delimiter) {
    std::string str;
    for (int dim = 0; dim < 4; dim++) {
        str.append(std::to_string(this->dims[dim]));
        str.append(delimiter);
    }
    str.pop_back();
    return str;
}

Tensor::Tensor(Dims d) : dims(d) {
    // size = product of all 4 dimensions
    if (d.Size() <= 0) {
        throw TensorDimsErr(d.GetSizeStr());
    }
    
    size_t size = this->dims.Size() * sizeof(float);
    
    this->data = (float*)malloc(size);
}

Tensor::Tensor(std::vector<int> d) : dims(d) {
    // size = product of all 4 dimensions
    size_t size = this->dims.Size() * sizeof(float);
    
    this->data = (float*)malloc(size);
}

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

void Tensor::LoadData(float* data) {
    this->data = data;
}
