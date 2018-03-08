kernel void Softmax(global float* input, global float* output) {
    float sum = 0;
    size_t id = get_global_id(0);
    size_t pos;
    for (pos = 0; pos < get_global_size(0); pos++) {
        sum += exp(input[pos]);
    }
    output[id] = exp(input[id]) / sum;
}

kernel void Backprop_Softmax(global float* input, global float* delta, global float* backprop) {
    float sum = 0;
    size_t id = get_global_id(0);
    size_t pos;
    for (pos = 0; pos < get_global_size(0); pos++) {
        if (pos != id) {
            sum += exp(input[pos]);
        }
    }
    float top = input[id] * sum;
    sum += input[id];
    float sftmx_deriv = top / (sum * sum);
    backprop[id] = sftmx_deriv * delta[id];
}
