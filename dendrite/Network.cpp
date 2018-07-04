//
//  Network.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "Network.hpp"
#include "Exceptions.hpp"

// Constructor (used in development, initialises blank network)
Network::Network(Tensor* i, Tensor* p, Tensor* o) {
    this->g = new Graph();
    this->g->LoadFixed();
    
    // learning F(i) -> o; p is output of network
    this->input = i;
    this->prediction = p;
    this->output = o;
    
    this->g->InitialiseLayers(this->input);
    
    // Initialise OpenCL dispatch queue
    this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    
    if (this->queue == NULL) {
        // Revert to CPU if GPU is unavailable
        this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
}

// Constructor, intialises network from a network save
Network::Network(std::string location) {
    // Open model.struct
    std::ifstream model_struct;
    std::string model_location = location;
    model_struct.open(model_location.append("model.struct"));
    
    // Initialise blank graph
    this->g = new Graph();
    
    // Intialise input tensor
    this->input = new Tensor(NetworkBufferParse::LoadInput(&model_struct));
    
    try {
        // Initialise OpenCL dispatch queue
        this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
        
        if (this->queue == NULL) {
            // Revert to CPU if GPU is unavailable
            this->queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
        }
    } catch (std::exception &e) {
        // Error initialising OpenCL dispatch queue (meaning that there is insufficient hardware)
        throw InsufficientHardware();
    }
    
    std::string line;
    GraphLoader::LayerDetails layer;
    
    // Load First Layer
    getline(model_struct, line);
    
    // Add the input layer to the graph
    g->InsertInput(this->input);
    
    // While the next Layer is a Layer
    while (GraphLoader::ParseLayer(line, &layer)) {
        g->InsertLayer(location, layer.type, layer.id, layer.inputs, layer.dependents);
        // Load next line
        getline(model_struct, line);
    }
    
    // Initialise all layers with weights, output and delta buffers
    this->g->InitialiseLayers(this->input);
    
    // Load learnable parameters from file
    this->g->LoadLayers(location);
    
    // Intialise prediction and output tensors
    this->prediction = new Tensor(this->g->GetOutputSize());
    this->output = new Tensor(this->g->GetOutputSize());
    
    // Implement Data Loading Methods
    std::string input_pipeline_location = location;
    std::string output_pipeline_location = location;
    input_pipeline_location.append("input_pipeline.instr");
    output_pipeline_location.append("output_pipeline.instr");
    
    // Intialise data pipeliens
    this->input_pipeline = new InstructionInterpreter(input_pipeline_location, this->input);
    this->output_pipeline = new InstructionInterpreter(output_pipeline_location, this->prediction);
    
    // Add loss function to graph
    this->g->AddLoss(GraphLoader::ParseLoss(line));
    
    model_struct.close();
}

void Network::Evaluate() {
    // Load data using IO pipelines
    this->input_pipeline->LoadNextDataBatch();
    this->output_pipeline->LoadNextDataBatch();
    
    // Evaluate network
    this->g->Evaluate(this->input, this->output, &(this->queue));
}

bool Network::Classify(float loss_threshold) {
    // Classify output
    return this->output_pipeline->Classify(this->output, this->prediction, loss_threshold);
}

float Network::Learn() {
    // Load data using IO pipelines
    this->input_pipeline->LoadNextDataBatch();
    this->output_pipeline->LoadNextDataBatch();
    
    // Perform one training iteration on graph
    return this->g->Learn(this->input, this->prediction, &(this->queue), this->LearningRate);
}

bool Network::SaveNetwork(std::string loc) {
    // Save network
    this->g->Save(loc, this->input->dims, this->prediction->dims);
    return true;
}

Tensor* Network::GetLayerData(int index) {
    // Return layer output
    return this->g->GetLayer(index, this->prediction);
}

Tensor* Network::GetLayerParams(int index) {
    // Return layer learnable parameters
    return this->g->GetLayerParams(index);
}

Dims Network::GetLayerDims(int index) {
    // Return layer output dimensions
    return this->g->GetLayer(index, this->prediction)->dims;
}
