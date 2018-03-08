kernel void ReLU(global float* input, global float* activ_map) {
    size_t pos = get_global_id(0);
    activ_map[pos] = fmax(float(0), input[pos]);
}

kernel void Backprop_ReLU(global float* input, global float* delta, global float* backprop) {
    size_t pos = get_global_id(0);
    if (input[pos] <= 0) {
        backprop[pos] = 0;
    } else {
        backprop[pos] = delta[pos];
    }
}
