kernel void FlattenError(global float* err_primary, global float* err) {
    size_t pos = get_global_id(0);
    err_primary[pos] += err[pos];
}
