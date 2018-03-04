//
//  Exceptions.cpp
//  dendrite
//
//  Created by Oli Callaghan on 02/03/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#include <stdio.h>
#include "Exceptions.hpp"

InsufficientArguments::InsufficientArguments(int p, int r) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Insufficient arguments passed, required at least " << r << ", passed " << p;
    this->err_msg = err_msg_strm.str();
}

const char* InsufficientArguments::what() const throw() {
    return this->err_msg.c_str();
}

ConversionError::ConversionError(std::string str, std::string to) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Error converting the string: " << str << ", to type: " << to;
    this->err_msg = err_msg_strm.str();
}

const char* ConversionError::what() const throw() {
    return this->err_msg.c_str();
}

UnknownMode::UnknownMode(std::string modes, std::string mode) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Unknown mode '" << mode << "', accepted modes are '" << modes;
    this->err_msg = err_msg_strm.str();
}

const char* UnknownMode::what() const throw() {
    return this->err_msg.c_str();
}

const char* InsufficientHardware::what() const throw() {
    return "Insufficient hardware to run network";
}

ModelStructSyntaxError::ModelStructSyntaxError(std::string str) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Syntax error in model struct: " << str;
    this->err_msg = err_msg_strm.str();
}

ModelStructSyntaxError::ModelStructSyntaxError(std::string str, std::string exp) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Syntax error in model struct: " << str << " expected " << exp;
    this->err_msg = err_msg_strm.str();
}

const char* ModelStructSyntaxError::what() const throw() {
    return this->err_msg.c_str();
}

UnsupportedLayerType::UnsupportedLayerType(std::string type) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Unsupported layer type: " << type;
    this->err_msg = err_msg_strm.str();
}

const char* UnsupportedLayerType::what() const throw() {
    return this->err_msg.c_str();
}

UnsupportedLossFunction::UnsupportedLossFunction(std::string lf) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Unsupported loss function " << lf;
    this->err_msg = err_msg_strm.str();
}

const char* UnsupportedLossFunction::what() const throw() {
    return err_msg.c_str();
}

FailedLoadingHP::FailedLoadingHP(short id, std::string t) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Failure loading hyperparameters for layer " << id << ", type: " << t;
    this->err_msg = err_msg_strm.str();
}

const char* FailedLoadingHP::what() const throw() {
    return err_msg.c_str();
}

FailedLoadingLP::FailedLoadingLP(short id, std::string t) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Failure loading learnable parameters for layer " << id << ", type: " << t;
    this->err_msg = err_msg_strm.str();
}

const char* FailedLoadingLP::what() const throw() {
    return this->err_msg.c_str();
}

FailedSavingHP::FailedSavingHP(short id) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Failure saving hyperparameters for layer " << id;
    this->err_msg = err_msg_strm.str();
}

const char* FailedSavingHP::what() const throw() {
    return err_msg.c_str();
}

FailedSavingLP::FailedSavingLP(short id) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Failure saving learnable parameters for layer " << id;
    this->err_msg = err_msg_strm.str();
}

const char* FailedSavingLP::what() const throw() {
    return this->err_msg.c_str();
}

IncorrectReadSize::IncorrectReadSize(long s, long exp_s) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Error reading " << s << " bytes into tensor of size " << exp_s;
    this->err_msg = err_msg_strm.str();
}

IncorrectReadSize::IncorrectReadSize(long s, long exp_s, bool) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Error reading " << s << " bits into tensor of size " << exp_s;
    this->err_msg = err_msg_strm.str();
}

const char* IncorrectReadSize::what() const throw() {
    return this->err_msg.c_str();
}

InvalidInstruction::InvalidInstruction(std::string instruction) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Invalid instruction: " << instruction;
    this->err_msg = err_msg_strm.str();
}

const char* InvalidInstruction::what() const throw() {
    return this->err_msg.c_str();
}

UnsupportedDataType::UnsupportedDataType(std::string type) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Unsupported data type: " << type;
    this->err_msg = err_msg_strm.str();
}

const char* UnsupportedDataType::what() const throw() {
    return this->err_msg.c_str();
}

GenericInstructionError::GenericInstructionError(std::string type) {
    this->err_msg = type;
}

const char* GenericInstructionError::what() const throw() {
    return this->err_msg.c_str();
}

GraphStructureError::GraphStructureError(std::string type) {
    this->err_msg = type;
}

const char* GraphStructureError::what() const throw() {
    return this->err_msg.c_str();
}

InvalidPARAMS::InvalidPARAMS(std::string params) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Invalid params: " << params;
    this->err_msg = err_msg_strm.str();
}

const char* InvalidPARAMS::what() const throw() {
    return this->err_msg.c_str();
}

TensorDimsErr::TensorDimsErr(std::string dims_str) {
    std::stringstream err_msg_strm;
    err_msg_strm << "Invalid dimensions: " << dims_str;
    this->err_msg = err_msg_strm.str();
}

const char* TensorDimsErr::what() const throw() {
    return this->err_msg.c_str();
}
