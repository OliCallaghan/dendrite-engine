//
//  InstructionInterpreter.cpp
//  dendrite
//
//  Created by Oli Callaghan on 30/01/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#include "InstructionInterpreter.hpp"

// Extract intial jump location from data pipeline
long GetJumpLocation(std::ifstream* file) {
    std::string line;
    getline(*file, line);
    
    std::regex loss_exprn("JUMP ([0-9]+)");
    std::smatch match;
    
    long pos = 0;
    
    if (std::regex_match(line, match, loss_exprn)) {
        // Parse position
        pos = atol(match[1].str().c_str());
    } else {
        throw GenericInstructionError("Second line of pipeline must be JUMP statement");
    }
    
    return pos;
}

// Extract repeat location from model.struct
long GetRepeatLocation(long start_pos, std::ifstream* file) {
    std::string line;
    getline(*file, line);
    
    std::regex loop_exp("LOOP FROM ([0-9]+) TO ([0-9]+)");
    std::smatch match;
    
    long s_pos = 0;
    long e_pos = 0;
    
    if (std::regex_match(line, match, loop_exp)) {
        // Parse repeat location
        s_pos = atol(match[1].str().c_str());
        e_pos = atol(match[2].str().c_str());
    } else {
        throw InvalidInstruction(line);
    }
    
    if (s_pos > e_pos) {
        throw InvalidInstruction(line);
    }
    
    if (start_pos != s_pos) {
        throw GenericInstructionError("LOOP start position must match JUMP position");
    }
    
    return e_pos;
}

// Convert data type string to data type
Pipeline::Data_T GetDataTypeFromStr(std::string type) {
    if (type == "UNSIGNED CHAR") {
        return Pipeline::Data_T::U_CHAR;
    } else if (type == "CHAR") {
        return Pipeline::Data_T::CHAR;
    } else if (type == "INT") {
        return Pipeline::Data_T::INT;
    } else if (type == "UNSIGNED INT") {
        return Pipeline::Data_T::U_INT;
    } else if (type == "FLOAT") {
        return Pipeline::Data_T::FLOAT;
    } else if (type == "BIT") {
        return Pipeline::Data_T::BIT;
    } else {
        throw UnsupportedDataType(type);
    }
}

// Convert data type to data type string
std::string GetStrFromDataType(Pipeline::Data_T type) {
    if (type == Pipeline::Data_T::U_CHAR) {
        return "UNSIGNED CHAR";
    } else if (type == Pipeline::Data_T::CHAR) {
        return "CHAR";
    } else if (type == Pipeline::Data_T::INT) {
        return "INT";
    } else if (type == Pipeline::Data_T::U_INT) {
        return "U_INT";
    } else if (type == Pipeline::Data_T::FLOAT) {
        return "FLOAT";
    } else if (type == Pipeline::Data_T::BIT) {
        return "BIT";
    } else if (type == Pipeline::Data_T::NONE) {
        return "NONE";
    } else {
        throw UnsupportedDataType("UNKNOWN");
    }
}

// Convert operation type to operation type string
std::string GetStrFromOp(Pipeline::Operation_T type) {
    switch (type) {
        case Pipeline::Operation_T::READ:
            return "READ";
            break;
        case Pipeline::Operation_T::ADD:
            return "ADD";
            break;
        case Pipeline::Operation_T::DIV:
            return "DIV";
            break;
        case Pipeline::Operation_T::MUL:
            return "MUL";
            break;
        case Pipeline::Operation_T::SOFTMAX:
            return "SOFTMAX";
        default:
            throw InvalidInstruction("UNKNOWN");
            break;
    }
}

// Load next instruction for data pipeline file
Pipeline::Operation* LoadNextIntruction(std::ifstream* file) {
    std::string line;
    getline(*file, line);
    
    std::regex read_rgx("READ ([0-9]+) AS (UNSIGNED CHAR|CHAR|INT|UNSIGNED INT|FLOAT|BIT)");
    std::regex mul_rgx("MUL (-?[0-9.]+)");
    std::regex div_rgx("DIV (-?[0-9.]+)");
    std::regex add_rgx("ADD (-?[0-9.]+)");
    std::regex softmax_rgx("SOFTMAX TO ([0-9]+)");
    std::regex repeat_rgx("REPEAT");
    std::smatch match;
    
    // Parse instruction
    if (std::regex_match(line, match, read_rgx)) {
        // READ instruction
        Pipeline::Data_T type = GetDataTypeFromStr(match[2].str());
        return new Pipeline::Operation(Pipeline::Operation_T::READ, atol(match[1].str().c_str()), type, NULL);
    } else if (std::regex_match(line, match, mul_rgx)) {
        // MULTIPLY
        return new Pipeline::Operation(Pipeline::Operation_T::MUL, NULL, Pipeline::Data_T::NONE, stof(match[1].str()));
    } else if (std::regex_match(line, match, div_rgx)) {
        // DIVIDE
        return new Pipeline::Operation(Pipeline::Operation_T::DIV, NULL, Pipeline::Data_T::NONE, stof(match[1].str()));
    } else if (std::regex_match(line, match, add_rgx)) {
        // ADD
        return new Pipeline::Operation(Pipeline::Operation_T::ADD, NULL, Pipeline::Data_T::NONE, stof(match[1].str()));
    } else if (std::regex_match(line, match, repeat_rgx)) {
        // NO MORE INSTRUCTIONS
        return NULL;
    } else if (std::regex_match(line, match, softmax_rgx)) {
        // CONVERT TO SOFTMAX DISTRIBUTION
        return new Pipeline::Operation(Pipeline::Operation_T::SOFTMAX, NULL, Pipeline::Data_T::NONE, stof(match[1].str()));
    } else {
        throw InvalidInstruction(line);
    }
}

// Print output instruction queue to console (used in development)
void InstructionInterpreter::OutputInstructionQueue() {
    for (int loc = 0; loc < this->op_list.size(); loc++) {
        std::cout << "[ ";
        std::cout << "OP: " << GetStrFromOp(this->op_list[loc].operation) << "\n";
        std::cout << "  DTYPE: " << GetStrFromDataType(this->op_list[loc].data_type) << "\n";
        if (this->op_list[loc].operation == Pipeline::Operation_T::READ) {
            std::cout << "  BYTES: " << this->op_list[loc].bytes << " ]";
        } else {
            std::cout << "  PARAM: " << this->op_list[loc].param << " ]";
        }
        std::cout << "\n";
    }
}

// Softmax classification
bool SOFTMAX_CF(Tensor* output, Tensor* should_output, float loss_threshold) {
    float max = 0;
    int max_pos = 0;
    for (int i = 0; i < output->dims.Size(); i++) {
        if (output->data[i] > max) {
            // Found maximum value in output tensor
            max = output->data[i];
            max_pos = i;
        }
    }
    
    if (should_output->data[max_pos] == 1) {
        // Check if maximum value in output tensor corresponds with value in exepcted output
        return true;
    }
    // Incorrect classification
    return false;
}

// L2 classification
bool L2_CF(Tensor* output, Tensor* should_output, float loss_threshold) {
    float loss_total = 0;
    for (int i = 0; i < output->dims.Size(); i++) {
        if (pow(output->data[i] - should_output->data[i], 2) > pow(loss_threshold,2)) {
            // Count number of output values which are within the threshold of the exepected output
            loss_total += 1;
        }
    }
    // Return the accuracy
    float loss = loss_total / output->dims.Size();
    if (loss < loss_threshold) {
        return true;
    }
    return false;
}

// Initialise data pipeline
InstructionInterpreter::InstructionInterpreter(std::string loc, Tensor* buf) : ten(buf), file(loc, std::ios::in) {
    std::string line;
    
    getline(this->file, line);
    
    // Load in Location of File
    this->handler = new BinaryFileHandler(line);
    
    long start_pos = GetJumpLocation(&this->file);
    this->handler->SetPosition(start_pos);
    
    long end_pos = GetRepeatLocation(start_pos, &this->file);
    this->handler->Loop(start_pos, end_pos);
    
    // Loop through until REPEAT
    Pipeline::Operation* op = LoadNextIntruction(&this->file);
    
    while (op != NULL) {
        this->op_list.push_back(*op);
        op = LoadNextIntruction(&this->file);
    }
    
    switch (this->op_list[this->op_list.size() - 1].operation) {
        case Pipeline::Operation_T::SOFTMAX:
            this->Classify = SOFTMAX_CF;
            break;
            
        default:
            this->Classify = L2_CF;
            break;
    }
}

InstructionInterpreter::~InstructionInterpreter() {
    this->file.close();
}

// READ instruction method
template <class T> void READ_F(BinaryFileHandler* handler, Tensor* buffer, long bytes, float param) {
    BinaryFileReader::ReadBytesToTensor<T>(handler, buffer, bytes);
}

// READ BIT instruction method
void READ_BIT(BinaryFileHandler* handler, Tensor* buffer, long bytes, float param) {
    // BYTES SHOULD ACTUALLY BE BITS
    BinaryFileReader::ReadBitsToTensor(handler, buffer, bytes);
}

// DIVIDE instruction method
void DIV_F(BinaryFileHandler* handler, Tensor* buffer, long bytes, float param) {
    for (int pos = 0; pos < buffer->dims.Size(); pos++) {
        buffer->data[pos] = buffer->data[pos] / param;
    }
}

// MULTIPLY instruction method
void MUL_F(BinaryFileHandler* handler, Tensor* buffer, long bytes, float param) {
    for (int pos = 0; pos < buffer->dims.Size(); pos++) {
        buffer->data[pos] = buffer->data[pos] * param;
    }
}

// ADD instruction method
void ADD_F(BinaryFileHandler* handler, Tensor* buffer, long bytes, float param) {
    for (int pos = 0; pos < buffer->dims.Size(); pos++) {
        buffer->data[pos] += param;
    }
}

// SOFTMAX instruction method
void SOFTMAX_F(BinaryFileHandler* handler, Tensor* buffer, long bytes, float param) {
    // FIRST ELEM = SOFTMAX PREDICTION
    int l = buffer->data[0];
    
    // Convert to softmax distribution
    for (int i = 0; i <= param; i++) {
        buffer->data[i] = 0;
    }
    buffer->data[l] = 1;
}

// Intiailise instruction
Pipeline::Operation::Operation(Pipeline::Operation_T i_t, long bytes, Data_T d_t, float param) {
    // Initialise Object
    this->operation = i_t;
    this->param = param;
    this->bytes = bytes;
    this->data_type = d_t;
    
    switch (i_t) {
        case Pipeline::Operation_T::DIV:
            // Set Execute Method
            this->Execute = DIV_F;
            break;
        case Pipeline::Operation_T::MUL:
            // Set Exec
            this->Execute = MUL_F;
            break;
        case Pipeline::Operation_T::ADD:
            // Set Exec Method
            this->Execute = ADD_F;
            break;
        case Pipeline::Operation_T::READ:
            // Set Exec Method
            switch (d_t) {
                case Pipeline::Data_T::CHAR:
                    // CHAR
                    this->Execute = READ_F<char>;
                    break;
                case Pipeline::Data_T::U_CHAR:
                    this->Execute = READ_F<unsigned char>;
                    break;
                case Pipeline::Data_T::INT:
                    this->Execute = READ_F<int>;
                    break;
                case Pipeline::Data_T::U_INT:
                    this->Execute = READ_F<unsigned int>;
                    break;
                case Pipeline::Data_T::FLOAT:
                    this->Execute = READ_F<float>;
                    break;
                case Pipeline::Data_T::BIT:
                    this->Execute = READ_BIT;
                    break;
                default:
                    break;
            }
            break;
        case Pipeline::Operation_T::SOFTMAX:
            // Set Softmax Distribution Conversion Execution
            this->Execute = SOFTMAX_F;
            break;
        default:
            // UNKNOWN OPERATION
            this->Execute = NULL;
            throw "UNKNOWN INSTRUCTION";
            break;
    }
}

// Load next data batch from file
void InstructionInterpreter::LoadNextDataBatch() {
    try {
        for (int instr = 0; instr < this->op_list.size(); instr++) {
            this->op_list[instr].Execute(handler, this->ten, this->op_list[instr].bytes, this->op_list[instr].param);
        }
    } catch (std::exception &e) {
        std::cerr << "Error loading data from pipeline\n";
        throw;
    }
}
