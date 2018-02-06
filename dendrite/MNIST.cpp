//
//  MNIST.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/12/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "MNIST.hpp"
#define LABEL_LOC "train-labels-idx1-ubyte"
#define IMAGE_LOC "train-IMAGES-idx3-ubyte"

MNISTHandler::MNISTHandler(std::string loc) : f_lab(loc + LABEL_LOC, std::ios::in | std::ios::binary), f_img(loc  + IMAGE_LOC, std::ios::in | std::ios::binary), location(loc) {
    this->pos_l = 8;
    this->pos_i = 16;
}

MNISTHandler::~MNISTHandler() {
    this->f_img.close();
    this->f_lab.close();
}

bool MNISTHandler::VerifyDataSet() {
    // really cant be bothered with verification atm
    f_img.seekg(this->pos_i);
    f_lab.seekg(this->pos_l);
    
    return true;
}

void MNISTHandler::LoadData(Tensor* input, Tensor* label) {
    // Return to start of dataset once epoch.
    if (this->pos_i >= 47040016) {
        this->pos_l = 8;
        this->pos_i = 16;
        f_img.seekg(this->pos_i);
        f_lab.seekg(this->pos_l);
    }
    char buffer[784];
    f_img.read(buffer,784);
    for (int i = 0; i < 784; i++) {
        input->data[i] = (float)static_cast<unsigned char>(buffer[i]) / 256;
    }
    
    this->pos_i += 784;
    this->pos_l += 1;
    
    char l_buf[1];
    f_lab.read(l_buf,1);
    for (int i = 0; i < 10; i++) {
        label->data[i] = 0;
    }
    label->data[l_buf[0]] = 1;
}

int MNISTHandler::Classify(Tensor* prediction) {
    float max = 0;
    int classify = 0;
    for (int pos = 0; pos < prediction->dims.SizePerEx(); pos++) {
        if (prediction->data[pos] > max) {
            max = prediction->data[pos];
            classify = pos;
        }
    }
    
    return classify;
}
