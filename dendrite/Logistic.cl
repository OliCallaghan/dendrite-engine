kernel void Logistic(global float* input, global float* activ_map) {
    size_t pos = get_global_id(0);
    activ_map[pos] = 1 / (1 + exp(-input[pos]));
}

kernel void Backprop(global float* input, global float* delta, global float* backprop) {
    size_t pos = get_global_id(0);
    backprop[pos] = (1 / (1 + exp(-input[pos])) * (1 - (1 / (1 + exp(-input[pos]))))) * delta[pos];
}
