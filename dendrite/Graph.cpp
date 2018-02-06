//
//  Graph.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "Graph.hpp"
#include <string>

Dims getDimsOfOutput(Dims input, Layers::Layer_T layer_t, void* hyperparameters) {
    switch (layer_t) {
        case Layers::FullyConnected_T:
            return Layers::FullyConnected::CalcOutputSize(input, *(Layers::FullyConnected::Hyperparameters*)hyperparameters);
            break;
            
        case Layers::Bias_T:
            return Layers::Bias::CalcOutputSize(input);
            break;
            
        case Layers::Logistic_T:
            return Layers::Logistic::CalcOutputSize(input);
            break;
            
        default:
            throw "Unsupported Layer";
            break;
    }
}

LearnableParameters* InitialiseLearnableParameters(Layers::Layer_T layer_t, void* hyperparameters, Dims dims) {
    switch (layer_t) {
        case Layers::FullyConnected_T:
            return Layers::FullyConnected::InitialiseLearnableParameters(*(Layers::FullyConnected::Hyperparameters*)hyperparameters, dims);
            break;
            
        case Layers::Bias_T:
            return Layers::Bias::InitialiseLearnableParameters(*(Layers::Bias::Hyperparameters*)hyperparameters, dims);
            
        default:
            throw "Unsupported Layer";
            break;
    }
    return NULL;
}

bool HasHyperparameters(Layers::Layer_T t) {
    switch (t) {
        case Layers::Layer_T::Logistic_T:
            return false;
            break;
        case Layers::Layer_T::FullyConnected_T:
            return true;
            break;
        case Layers::Layer_T::Bias_T:
            return true;
            break;
        default:
            throw "Unknown Layer Type";
            break;
    }
}

bool Graph::Load(std::string location, Tensor* input) {
    std::ifstream model_struct;
    model_struct.open(location.append("/model.struct"));
    
    std::string line;
    std::vector<std::string> laybuf = {};
    
    // Import each line
    while (getline(model_struct, line)) {
        laybuf.push_back(line);
    }
    
    if (laybuf.size() < 3) {
        // Invalid network size
        throw "Invalid network size.";
    }
    
    // Process input
    // laybuf[0];
    // GraphLoader::ParseLine(laybuf[0]);
    /*this->layers.push_back(*GraphLoader::ParseInput(laybuf[0], input));
    
    for (int lay = 1; lay < laybuf.size() - 1; lay++) {
        this->layers.push_back(*GraphLoader::ParseLayer(laybuf[lay], location));
    }
    
    this->layer_n = this->layers.size();
    */
    // Interpret final loss layer
    //this->loss_fn = new Loss::LossFn(GraphLoader::ParseLoss(laybuf[laybuf.size() - 1]));
    
    // Implement loading
    return true;
}

bool Graph::LoadFixed() {
    std::vector<short> i0 = {NULL};
    std::vector<short> d0 = {1};
    this->layers.push_back(*new Layer(Layers::Layer_T::Input_T, i0, d0, NULL));
    
    // FC Layer 1
    std::vector<short> i1 = {0};
    std::vector<short> d1 = {2};
    Layers::FullyConnected::Hyperparameters* h_p1 = new Layers::FullyConnected::Hyperparameters(300);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i1, d1, (void*)(h_p1)));
    
    // Bias Activation
    std::vector<short> i2 = {1};
    std::vector<short> d2 = {3};
    Layers::Bias::Hyperparameters* h_p2 = new Layers::Bias::Hyperparameters(0,0.1);
    this->layers.push_back(*new Layer(Layers::Layer_T::Bias_T, i2, d2, (void*)(h_p2)));
    
    // FC Layer 2
    std::vector<short> i3 = {2};
    std::vector<short> d3 = {4};
    Layers::FullyConnected::Hyperparameters* h_p3 = new Layers::FullyConnected::Hyperparameters(300);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i3, d3, (void*)(h_p3)));
    
    // Logistic Activation
    std::vector<short> i4 = {3};
    std::vector<short> d4 = {5};
    this->layers.push_back(*new Layer(Layers::Layer_T::Logistic_T, i4, d4, NULL));
    
    // FC Layer 3
    std::vector<short> i5 = {4};
    std::vector<short> d5 = {6};
    Layers::FullyConnected::Hyperparameters* h_p5 = new Layers::FullyConnected::Hyperparameters(10);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i5, d5, (void*)(h_p5)));
    
    std::vector<short> i6 = {5};
    std::vector<short> d6 = {6};
    this->layers.push_back(*new Layer(Layers::Layer_T::Logistic_T, i6, d6, NULL));
    
    /*
    std::vector<short> i2 = {1};
    std::vector<short> d2 = {3};
    Layers::FullyConnected::Hyperparameters* h_p2 = new Layers::FullyConnected::Hyperparameters(100);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i2, d2, (void*)(h_p2)));
    
    std::vector<short> i3 = {2};
    std::vector<short> d3 = {3};
    Layers::FullyConnected::Hyperparameters* h_p3 = new Layers::FullyConnected::Hyperparameters(10);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i3, d3, (void*)(h_p3)));*/
    
    //std::vector<short> i3 = {2};
    //std::vector<short> d3 = {NULL};
    //Layers::Bias::Hyperparameters* h_p3 = new Layers::Bias::Hyperparameters(0, 0.1);
    //this->layers.push_back(*new Layer(Layers::Layer_T::Bias_T, i3, d3, h_p3));
    
    this->layer_n = this->layers.size();
    this->loss_t = Loss::L2_T;
    this->loss_fn = new Loss::LossFn(Loss::L2_T);
    
    return true;
}

// Returns Correct Hyperparameter object
void* ExtractHyperparameters(std::string loc, short id, Layers::Layer_T type) {
    std::ifstream hyperparameters_f;
    // Determine file name
    std::stringstream full_location;
    full_location << loc << "hparams/hp" << id << ".dat";
    
    // Open Hyperparameters file
    hyperparameters_f.open(full_location.str(), std::ios::out | std::ios::binary);
    
    size_t size;
    
    switch (type) {
        case Layers::FullyConnected_T:
            size = sizeof(Layers::FullyConnected::Hyperparameters);
            break;
        case Layers::Bias_T:
            size = sizeof(Layers::Bias::Hyperparameters);
            break;
        default:
            break;
    }
    
    void* hp = malloc(size);
    
    hyperparameters_f.read((char*)hp, size);
    
    return hp;
}

bool Graph::InsertLayer(std::string loc, Layers::Layer_T type, short id, std::vector<short> inputs, std::vector<short> dependents) {
    void* hp = NULL;
    if (HasHyperparameters(type)) {
        hp = ExtractHyperparameters(loc, id, type);
    }
    this->layers.push_back(*new Layer(type, inputs, dependents, hp));
    this->layers[this->layers.size() - 1];
    
    this->layer_n = this->layers.size();
    
    return true;
}

bool Graph::InsertInput(Tensor* input) {
    this->layers.push_back(*new Layer(Layers::Layer_T::Input_T, {NULL}, {1}, NULL));
    this->layer_n = this->layers.size();
    return true;
}

bool Graph::InitialiseLayers(Tensor* input) {
    // Check that layer 0 is input
    if (this->layers[0].layer_t != Layers::Layer_T::Input_T) {
        throw "ERR: First layer must be input layer!";
    }
    
    this->layers[0].output = input;
    
    for (int pos = 1; pos < this->layer_n; pos++) {
        // Initialise output buffer
        this->layers[pos].output = new Tensor(getDimsOfOutput(this->layers[this->layers[pos].input[0]].output->dims, this->layers[pos].layer_t, this->layers[pos].hyperparameters));
        // Initialise delta buffer
        this->layers[pos].delta = new Tensor(getDimsOfOutput(this->layers[this->layers[pos].input[0]].output->dims, this->layers[pos].layer_t, this->layers[pos].hyperparameters));
        // Initialise learnable parameters
        if (this->layers[pos].has_params == true) {
            this->layers[pos].params = InitialiseLearnableParameters(this->layers[pos].layer_t, this->layers[pos].hyperparameters, this->layers[this->layers[pos].input[0]].output->dims);
        }
    }
    
    return true;
}

void Graph::Evaluate(Tensor* input, Tensor* output, dispatch_queue_t* queue) {
    for (size_t pos = 1; pos < this->layer_n; pos++) {
        // Iterate over every layer
        Tensor* inputs[this->layers[pos].input.size()];
        for (int inp = 0; inp < this->layers[pos].input.size(); inp++) {
            inputs[inp] = this->layers[this->layers[pos].input[inp]].output;
        }
        this->layers[pos].ForwardFunc(inputs, this->layers[pos].output, this->layers[pos].params, this->layers[pos].hyperparameters, queue);
    }
    
    output->data = this->layers[this->layer_n - 1].output->data;
}

float Graph::Learn(Tensor* input, Tensor* prediction, dispatch_queue_t* queue, float eta) {
    for (size_t pos = 1; pos < this->layer_n; pos++) {
        // Iterate over every layer
        Tensor* inputs[this->layers[pos].input.size()];
        for (int inp = 0; inp < this->layers[pos].input.size(); inp++) {
            inputs[inp] = this->layers[this->layers[pos].input[inp]].output;
        }
        this->layers[pos].ForwardFunc(inputs, this->layers[pos].output, this->layers[pos].params, this->layers[pos].hyperparameters, queue);
    }
    
    // Calculate loss for final layer
    this->loss_fn->Loss(this->layers[this->layer_n - 1].output, prediction, this->layers[this->layer_n - 1].delta, queue);
    
    for (size_t pos = this->layer_n - 1; pos > 1; pos--) {
        // Collect array of dependents
        Tensor* dependents[this->layers[pos - 1].dependents.size()];
        
        for (int dpt = 0; dpt < this->layers[pos - 1].dependents.size(); dpt++) {
            dependents[dpt] = this->layers[this->layers[pos].dependents[dpt]].delta;
        }
        
        // Backpropagate the error
        this->layers[pos].BackpropDeltasFunc(dependents, this->layers[pos - 1].delta, this->layers[this->layers[pos].input[0]].output ,this->layers[pos].params, this->layers[pos].hyperparameters, queue);
        
        // Update weights
        this->layers[pos].CalcParamDeltasFunc(this->layers[pos].delta, this->layers[this->layers[pos].input[0]].output, this->layers[pos].params, this->layers[pos].hyperparameters, eta, queue);
    }
    
    this->layers[1].CalcParamDeltasFunc(this->layers[1].delta, this->layers[this->layers[1].input[0]].output, this->layers[1].params, this->layers[1].hyperparameters, eta, queue);
    
    float loss = this->loss_fn->Loss_Val(this->layers[this->layer_n - 1].output, prediction, queue);
    
    return loss;
}

std::string ReturnLayerStr(Layers::Layer_T type) {
    switch (type) {
        case Layers::FullyConnected_T:
            return "FC";
            break;
        
        case Layers::Logistic_T:
            return "LG";
            break;
            
        case Layers::Bias_T:
            return "B";
            break;
            
        default:
            throw "Unsupported Layer in Saving File";
            break;
    }
}

std::string ReturnVectorStr(std::vector<short> vec) {
    std::stringstream vec_str;
    for (int elem = 0; elem < vec.size(); elem++) {
        vec_str << vec[elem] << ",";
    }
    std::string vec_str_final = vec_str.str();
    vec_str_final.pop_back();
    
    return vec_str_final;
}

std::string ReturnLossStr(Loss::Loss_T type) {
    switch (type) {
        case Loss::L2_T:
            return "L2";
            break;
            
        default:
            break;
    }
}

bool Graph::Save(std::string loc, Dims in, Dims out) {
    std::string model_loc = loc;
    model_loc.append("/model.struct");
    std::ofstream file(model_loc);
    std::stringstream buf;
    buf << "<inp s=" << in.GetSizeStr(",") << ">\n";
    for (int lay = 1; lay < this->layers.size(); lay++) {
        std::stringstream layer_str;
        layer_str << "<lay t=" << ReturnLayerStr(this->layers[lay].layer_t) << " id=" << lay << " i=" << ReturnVectorStr(this->layers[lay].input) << " d=" << ReturnVectorStr(this->layers[lay].dependents) << ">\n";
        buf << layer_str.str();
    }
    buf << "<loss f=" << ReturnLossStr(this->loss_t) << ">\n";
    buf << "<out s=" << out.GetSizeStr(",") << ">\n";
    file << buf.str();
    
    // Trigger save on each layer
    
    for (int lay = 1; lay < this->layers.size(); lay++) {
        if (this->layers[lay].has_params) {
            this->layers[lay].SaveLearnableParameters(loc, lay);
            this->layers[lay].SaveHyperparameters(loc, lay);
        }
    }
    
    file.close();
    
    return true;
}

bool Graph::LoadLayers(std::string loc) {
    for (int lay = 1; lay < this->layers.size(); lay++) {
        if (this->layers[lay].has_params) {
            this->layers[lay].LoadLearnableParameters(loc, lay);
        }
    }
    return true;
}

Dims Graph::GetOutputSize() {
    return this->layers[layer_n - 1].output->dims;
}

void Graph::AddLoss(Loss::Loss_T loss_t) {
    switch (loss_t) {
        case Loss::L2_T:
            this->loss_t = Loss::L2_T;
            this->loss_fn = new Loss::LossFn(Loss::L2_T);
            break;
            
        default:
            throw "Unsupported Loss";
            break;
    }
}
