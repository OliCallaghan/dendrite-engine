//
//  Exceptions.hpp
//  dendrite
//
//  Created by Oli Callaghan on 02/03/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#ifndef Exceptions_hpp
#define Exceptions_hpp

#include <stdio.h>
#include <exception>
#include <sstream>

// Insufficient Arguments Error
class InsufficientArguments: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    InsufficientArguments(int, int);
};

// Conversion Error
class ConversionError: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    ConversionError(std::string, std::string);
};

// Unknown Execution Mode Error
class UnknownMode: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    UnknownMode(std::string, std::string);
};

// Insufficient Hardware Available Error
class InsufficientHardware: public std::exception {
public:
    virtual const char* what() const throw();
};

// Zero Size Dims Error
class TensorDimsErr: public std::exception {
    std::string err_msg;
public:
    TensorDimsErr(std::string);
    virtual const char* what() const throw();
};

// Invalid PARAMS Syntax Error
class InvalidPARAMS: public std::exception {
    std::string err_msg;
public:
    InvalidPARAMS(std::string);
    virtual const char* what() const throw();
};

// Syntax Error in model.struct Error
class ModelStructSyntaxError: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    ModelStructSyntaxError(std::string);
    ModelStructSyntaxError(std::string, std::string);
};

// Unsupported Layer Type Error
class UnsupportedLayerType: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    UnsupportedLayerType(std::string);
};

// Unsupported Loss Function Error
class UnsupportedLossFunction: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    UnsupportedLossFunction(std::string);
};

// Failure Loading Hyperparameters Error
class FailedLoadingHP: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    FailedLoadingHP(short, std::string);
};

// Failure Loading Learnable Parameters Error
class FailedLoadingLP: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    FailedLoadingLP(short, std::string);
};

class FailedSavingHP: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    FailedSavingHP(short);
};

class FailedSavingLP: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    FailedSavingLP(short);
};

class IncorrectReadSize: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    IncorrectReadSize(long, long);
    IncorrectReadSize(long, long, bool);
};

// Invalid Instruction Error
class InvalidInstruction: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    InvalidInstruction(std::string);
};

// Unsupported Data Type Error
class UnsupportedDataType: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    UnsupportedDataType(std::string);
};

// Generic Instruction Error
class GenericInstructionError: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    GenericInstructionError(std::string);
};

// Graph Structure Error
class GraphStructureError: public std::exception {
    std::string err_msg;
public:
    virtual const char* what() const throw();
    GraphStructureError(std::string);
};

#endif /* Exceptions_hpp */
