#ifndef __COMMON_H__
#define __COMMON_H__

#include <string>

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

template <typename T>
bool allocate_memory_on_device(T **arr, std::string arr_name, int count) {
    size_t memory_size = count * sizeof(T);
    // cout << "\nAllocating " << memory_size << " bytes for " << arr_name << " on device..." << endl;

    cudaError_t err = cudaMalloc((void **) arr, memory_size); // allocate global memory on the device
    if (err != cudaSuccess) {
        // cout << "error allocating memory for " << arr_name << " on device: " << memory_size << " bytes" << endl;
        return false;
    } else {
        gpuErrchk(cudaMemset(*arr, 0, memory_size));
        // cout << "device: memory for " << arr_name << " allocated successfully." << endl;
        return true;
    }
}

struct forest_t {
    int* decision_trees = nullptr;
    int* leaf_class = nullptr;
    int* leaf_back = nullptr;
    int* leaf_id_range_end = nullptr;
    int* leaf_counters = nullptr;
    int* samples_seen_count = nullptr;
    int* tree_confusion_matrices = nullptr;
};

struct tree_t {
    int* tree = nullptr;
    int* leaf_class = nullptr;
    int* leaf_back = nullptr;
    int* leaf_id_range_end = nullptr;
    int* leaf_counter = nullptr;
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

double get_kappa(int *confusion_matrix, int class_count, double accuracy, int sample_count);

#endif
