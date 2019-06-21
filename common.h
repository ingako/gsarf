#ifndef COMMON_H
#define COMMON_H

extern bool STATE_ADAPTIVE; 

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
extern long ALL_LEAF_COUNTERS_SIZE;

extern float r;
extern float delta;

extern double warning_delta;
extern double drift_delta;

extern double kappa_threshold;
extern double bg_tree_add_delta;
extern int CPU_TREE_POOL_SIZE;

#define EPS 1e-9
#define IS_BIT_SET(val, pos) (val & (1 << pos))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct tree_t {
    int* tree = nullptr;
    int* leaf_class = nullptr;
    int* leaf_back = nullptr;
    int* leaf_id_range_end = nullptr;
    int* leaf_counter = nullptr;
    int* cur_node_count_per_tree = nullptr;
    int* cur_leaf_count_per_tree = nullptr;
    int* samples_seen_count = nullptr;
};

struct candidate_t {
    int tree_id;
    int forest_idx;
    double kappa;

    candidate_t(int t, int f) {
        tree_id = t;
        forest_idx = f;
    }

    bool operator<(const candidate_t& a) const {
        return kappa < a.kappa;
    }
};

double get_kappa(int *confusion_matrix, int class_count, double accuracy, int sample_count) {
    // computes the Cohen's kappa coefficient

    double p0 = accuracy;
    double pc = 0.0;
    int row_count = class_count;
    int col_count = class_count;

    for (int i = 0; i < row_count; i++) {
        double row_sum = 0;
        for (int j = 0; j < col_count; j++) {
            row_sum += confusion_matrix[i * col_count + j];
        }

        double col_sum = 0;
        for (int j = 0; j < row_count; j++) {
            col_sum += confusion_matrix[j * row_count + i];
        }

        pc += (row_sum / sample_count) * (col_sum / sample_count);
    }

    if (pc == 1) {
        return 1;
    }

    return (p0 - pc) / (1.0 - pc);
}

#endif
