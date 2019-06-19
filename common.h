#ifndef COMMON_H
#define COMMON_H

extern int TREE_COUNT;
extern int TREE_DEPTH_PARAM;
extern int INSTANCE_COUNT_PER_TREE;
extern int SAMPLE_FREQUENCY;
extern int CLASS_COUNT;

extern int FOREGROUND_TREE_COUNT;
extern int GROWING_TREE_COUNT;

extern int ATTRIBUTE_COUNT_TOTAL;
extern int ATTRIBUTE_COUNT_PER_TREE;
extern int TREE_DEPTH;

extern int NODE_COUNT_PER_TREE;
extern int LEAF_COUNT_PER_TREE;

extern int LEAF_COUNTER_SIZE;
extern int LEAF_COUNTERS_SIZE_PER_TREE;

extern float r;
extern float delta;

#define EPS 1e-9
#define IS_BIT_SET(val, pos) (val & (1 << pos))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#endif
