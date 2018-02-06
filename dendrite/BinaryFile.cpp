//
//  BinaryFile.cpp
//  dendrite
//
//  Created by Oli Callaghan on 29/01/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#include "BinaryFile.hpp"

BinaryFileHandler::BinaryFileHandler(std::string loc) : file(loc, std::ios::in | std::ios::binary), location(loc) {
    this->file.seekg(0);
}

BinaryFileHandler::~BinaryFileHandler() {
    this->file.close();
}

void BinaryFileHandler::SetPosition(long pos) {
    this->file.seekg(pos);
}

void BinaryFileHandler::Loop(long start_pos, long end_pos) {
    this->start_pos = start_pos;
    this->end_pos = end_pos;
}

void BinaryFileHandler::Increment(long n) {
    if (static_cast<long>(this->file.tellg()) + n >= end_pos) {
        this->file.seekg(this->start_pos);
    }
}

/*template <class T> void BinaryFileReader::ReadBytesToTensor(BinaryFileHandler* handler, Tensor* buffer, long bytes_n) {
    // Read file and loop
    handler->Increment(sizeof(T) * buffer->dims.Size());
    T bytes[buffer->dims.Size()];
    handler->file.read(reinterpret_cast<char*>(bytes), sizeof(T) * buffer->dims.Size());
    
    for (int i = 0; i < buffer->dims.Size(); i++) {
        buffer->data[i] = bytes[i];
    }
}*/

void BinaryFileReader::ReadBitsToTensor(BinaryFileHandler* handler, Tensor* buffer, long bits) {
    
    long len = (bits + 7) / 8;
    
    char dat[len];
    
    // Read file and loop
    handler->Increment(len);
    handler->file.read(dat, len);
    
    for (int i = 0; i < buffer->dims.Size(); i++) {
        buffer->data[i] = (dat[i / 8] >> (i % 8)) & 1;
    }
}
