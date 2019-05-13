#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <map>
#include <iomanip>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "ADWIN.cu"
#include "LRU_state.cu"
#include "state_graph.cu"

using namespace std;

#define EPS 1e-10
#define IS_BIT_SET(val, pos) (val & (1 << pos))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

static int NODE_COUNT_PER_TREE;
static int LEAF_COUNT_PER_TREE;
static int LEAF_COUNTERS_SIZE_PER_TREE;

struct tree_t {
    int* tree;
    int* leaf_class;
    int* leaf_back;
    int* leaf_counter;
    int* cur_node_count_per_tree;
    int* cur_leaf_count_per_tree;
    int* samples_seen_count;
    int* confusion_matrix;
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

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <typename T>
bool allocate_memory_on_device(T **arr, string arr_name, int count) {
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
            col_sum += confusion_matrix[i * row_count + j];
        }

        pc += (row_sum / sample_count) * (col_sum / sample_count);
    }

    if (pc == 1) {
        return 1;
    }

    return (p0 - pc) / (1.0 - pc);
}



void select_k_attributes(int *reservoir, int n, int k) {
    for (int i = 0; i < k; i++) {
        reservoir[i] = rand() % n;
    }

    // int i;
    // for (i = 0; i < k; i++) {
    //     reservoir[i] = i;
    // }

    // for (i = k; i < n; i++) {
    //     int j = rand() % (i + 1);

    //     if (j < k) reservoir[j] = i;
    // }
}

vector<string> split_attributes(string line, char delim) {
    vector<string> arr;
    const char *start = line.c_str();
    bool instring = false;

    for (const char* p = start; *p; p++) {
        if (*p == '"') {
            instring = !instring;
        } else if (*p == delim && !instring) {
            arr.push_back(string(start, p-start));
            start = p + 1;
        }
    }

    arr.push_back(string(start)); // last field delimited by end of line instead of comma
    return arr;
}


vector<string> split(string str, string delim) {
    char* cstr = const_cast<char*>(str.c_str());
    char* current;
    vector<string> arr;
    current = strtok(cstr, delim.c_str());

    while (current != NULL) {
        arr.push_back(current);
        current = strtok(NULL, delim.c_str());
    }

    return arr;
}

void tree_memcpy(tree_t *from_tree, tree_t *to_tree, bool is_background_tree) {

    memcpy(to_tree->tree, from_tree->tree, NODE_COUNT_PER_TREE * sizeof(int));

    memcpy(to_tree->leaf_class, from_tree->leaf_class, LEAF_COUNT_PER_TREE * sizeof(int));

    memcpy(to_tree->leaf_back, from_tree->leaf_back, LEAF_COUNT_PER_TREE * sizeof(int));

    if (is_background_tree) {
        memcpy(to_tree->leaf_counter, from_tree->leaf_counter,
                LEAF_COUNTERS_SIZE_PER_TREE * sizeof(int));

        memcpy(to_tree->cur_node_count_per_tree,
                from_tree->cur_node_count_per_tree,
                sizeof(int));

        memcpy(to_tree->cur_leaf_count_per_tree,
                from_tree->cur_leaf_count_per_tree,
                sizeof(int));

        memcpy(to_tree->samples_seen_count,
                from_tree->samples_seen_count,
                LEAF_COUNT_PER_TREE * sizeof(int));
    }
}


__global__ void setup_kernel(curandState *state) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(42, idx, 0, &state[idx]);
}

__device__ unsigned int get_left(unsigned int index) {
    return 2 * index + 1;
}

__device__ unsigned int get_right(unsigned int index) {
    return 2 * index + 2;
}

__device__ int get_rand(int low, int high, curandState *local_state) {
    float randu_f = curand_uniform(local_state);
    randu_f *= (high - low + 0.999999);
    randu_f += low;
    int randu_int = __float2int_rz(randu_f);

    return randu_int;
}

__device__ int poisson(float lambda, curandState *local_state) {
	float product = 1.0;
	float sum = 1.0;

    int rand_num = get_rand(0, 1000, local_state);

	float next_double = (float) rand_num / 1000.0;
	float threshold = next_double * exp(lambda);
	int max_val = max(100, 10 * (int) (lambda));

	int i = 1;
	while ((i < max_val) && (sum <= threshold)) {
		product *= (lambda / i);
		sum += product;
		i++;
	}

	return i - 1;
}

__global__ void reset_tree(
        int *reseted_tree_idx_arr,
        int *decision_trees,
        int *leaf_counters,
        int *leaf_class,
        int *leaf_back,
        int *samples_seen_count,
        int *node_count_per_tree,
        int *leaf_count_per_tree,
        int *tree_confusion_matrix,
        int max_node_count_per_tree,
        int max_leaf_count_per_tree,
        int leaf_counter_size,
        int leaf_counter_row_len,
        int confusion_matrix_size,
        int class_count) {

    // <<<1, reseted_tree_count>>>

    if (threadIdx.x >= blockDim.x) {
        return;
    }

    int tree_idx = reseted_tree_idx_arr[threadIdx.x];

    node_count_per_tree[tree_idx] = 1;
    leaf_count_per_tree[tree_idx] = 1;

    int *cur_decision_tree = decision_trees + tree_idx * max_node_count_per_tree;
    int *cur_leaf_class = leaf_class + tree_idx * max_leaf_count_per_tree;
    int *cur_leaf_back = leaf_back + tree_idx * max_leaf_count_per_tree;
    int *cur_samples_seen_count = samples_seen_count + tree_idx * max_leaf_count_per_tree;

    cur_decision_tree[0] = (1 << 31);
    cur_leaf_class[0] = 0;
    cur_leaf_back[0] = 0;

    for (int i = 0; i < max_leaf_count_per_tree; i++) {
        cur_samples_seen_count[i] = 0;
    }

    int *cur_leaf_counter = leaf_counters + tree_idx * max_leaf_count_per_tree * leaf_counter_size;

    for (int k = 0; k < class_count + 2; k++) {
        for (int ij = 0; ij < leaf_counter_row_len; ij++) {
            cur_leaf_counter[k * leaf_counter_row_len + ij] = k == 1 ? 1 : 0;
        }
    }

    // int *cur_tree_confusion_matrix = tree_confusion_matrix + tree_idx * confusion_matrix_size;

    // for (int i = 0; i < confusion_matrix_size; i++) {
    //     cur_tree_confusion_matrix[i] = 0;
    // }
}

__global__ void tree_traversal(
        int *decision_trees,
        int *tree_status,
        int *data,
        int *reached_leaf_ids,
        int *is_leaf_active,
        int *leaf_class,
        int *correct_counter,
        int *samples_seen_count,
        int *forest_vote,
        int *forest_vote_idx_arr,
        int *weights,
        int *tree_error_count,
        int *confusion_matrix,
        int *tree_confusion_matrix,
        int *class_count_arr,
        int majority_class,
        int node_count_per_tree,
        int leaf_count_per_tree,
        int attribute_count_total,
        int class_count,
        int confusion_matrix_size,
        curandState *state) {
    // <<<TREE_COUNT, INSTANCE_COUNT_PER_TREE>>>

    int tree_idx = blockIdx.x;

    int cur_tree_status = tree_status[tree_idx];
    if (cur_tree_status == 0 || cur_tree_status == 2 || cur_tree_status == 4) {
        // tree is inactive
        // or an inactive background tree
        // or an empty candidate tree
        return;
    }

    int instance_idx = threadIdx.x;
    int instance_count_per_tree = blockDim.x;
    int thread_pos = instance_idx + tree_idx * instance_count_per_tree;

    if (thread_pos >= blockDim.x * gridDim.x) {
        return;
    }

    int *cur_data_line = data + instance_idx * (attribute_count_total + 1);
    int *cur_decision_tree = decision_trees + tree_idx * node_count_per_tree;
    int *cur_leaf_class = leaf_class + tree_idx * leaf_count_per_tree;
    int *cur_samples_seen_count = samples_seen_count + tree_idx * leaf_count_per_tree;
    int *cur_forest_vote = forest_vote + instance_idx * class_count;
    int *cur_tree_confusion_matrix = tree_confusion_matrix + tree_idx * confusion_matrix_size;

    int pos = 0;
    while (!IS_BIT_SET(cur_decision_tree[pos], 31)) {
        int attribute_id = cur_decision_tree[pos];
        pos = cur_data_line[attribute_id] == 0 ? get_left(pos) : get_right(pos);
    }

    if (pos >= node_count_per_tree) {
            printf("pos out of node_count_per_tree: %i\n", tree_idx);
    }

    if (cur_decision_tree[pos] == -1) {
            printf("cannot be -1: %i\n", tree_idx);
    }

    int leaf_offset = (cur_decision_tree[pos] & (~(1 << 31)));

    atomicAdd(&cur_samples_seen_count[leaf_offset], 1);

    // online bagging
    int *cur_weights = weights + tree_idx * instance_count_per_tree;

    // curand library poisson is super slow!
    // cur_weights[instance_idx] = curand_poisson(state + thread_pos, 1.0);

    // prepare weights to be used in counter_increase kernel
    cur_weights[instance_idx] = poisson(1.0, state + thread_pos);
    // printf("==================================cur weight: %i\n", cur_weights[instance_idx]);

    int predicted_class = cur_leaf_class[leaf_offset];
    int actual_class = cur_data_line[attribute_count_total];

    if (pos == 0) {
        predicted_class = majority_class;
    }

    if (predicted_class != actual_class) {
        atomicAdd(&tree_error_count[tree_idx], 1);
    }

    atomicAdd(&cur_tree_confusion_matrix[actual_class * class_count + predicted_class], 1);

    if (cur_tree_status == 5) {
        return;
    }

    int *cur_reached_leaf_ids = reached_leaf_ids + tree_idx * instance_count_per_tree;
    cur_reached_leaf_ids[instance_idx] = leaf_offset;

    int *cur_is_leaf_active = is_leaf_active + tree_idx * leaf_count_per_tree;
    cur_is_leaf_active[leaf_offset] = 1;

    if (cur_tree_status == 3) {
        // growing background tree does not particiate in voting
        return;
    }

    if (class_count_arr[predicted_class] == 0) {
        predicted_class = majority_class;
    }

    atomicAdd(&cur_forest_vote[predicted_class], 1);

    if (tree_idx != 0) {
        return;
    }

    int *cur_forest_vote_idx_arr = forest_vote_idx_arr + instance_idx * class_count;

    thrust::sort_by_key(thrust::seq,
            cur_forest_vote,
            cur_forest_vote + class_count,
            cur_forest_vote_idx_arr);

    int voted_class = cur_forest_vote_idx_arr[class_count - 1];

    atomicAdd(&confusion_matrix[actual_class * class_count + voted_class], 1);

    if (voted_class == actual_class) {
        atomicAdd(correct_counter, 1);
    }
}

__global__ void counter_increase(
        int *leaf_counters,
        int *tree_status,
        int *reached_leaf_ids,
        int *data,
        int *weights,
        int class_count,
        int attribute_count_total,
        int leaf_count_per_tree,
        int leaf_counter_size) {
    // gridDim: dim3(TREE_COUNT, INSTANCE_COUNT_PER_TREE)
    // blockDim: ATTRIBUTE_COUNT_TOTAL
    // increment both n_ij (at row 0) and n_ijk (at row k)

    // input: an array of leaf_ids (offset) and leaf_classes built from tree_traversal

    // *** Each leaf counter is represented by a block and uses one thread for each attribute i and
    // value j (i.e. one thread per column)
    //
    // Row 0 stores the total number of times value n_ij appeared.
    // Row 1 is a mask that keeps track of which attributes have been already used in internal nodes
    // along the path.
    // Row 2 and onwards stores partial counters n_ijk for each class k.

    int tree_idx = blockIdx.x;
    int cur_tree_status = tree_status[tree_idx];
    if (cur_tree_status == 0 || cur_tree_status == 2) {
        return;
    }

    int instance_idx = blockIdx.y;
    int instance_count_per_tree = gridDim.y;

    int block_id = blockIdx.y + blockIdx.x * gridDim.y;

    int thread_pos = threadIdx.x + block_id * blockDim.x;
    if (thread_pos >= gridDim.x * gridDim.y * blockDim.x) {
        return;
    }

    int *cur_reached_leaf_ids = reached_leaf_ids + tree_idx * instance_count_per_tree;
    int reached_leaf_id = cur_reached_leaf_ids[instance_idx];

    int *cur_data = data + instance_idx * (attribute_count_total + 1);
    int *cur_weights = weights + tree_idx * instance_count_per_tree;
    int cur_weight = cur_weights[instance_idx];

    // the counter start position corresponds to the leaf_id i.e. leaf offset
    int counter_start_pos = reached_leaf_id * leaf_counter_size + tree_idx *
        leaf_count_per_tree * leaf_counter_size;
    int *cur_leaf_counter = leaf_counters + counter_start_pos;
    // printf("leaf counter start pos is:  %i\n", counter_start_pos);

    int ij = cur_data[threadIdx.x] + threadIdx.x * 2; // binary value 0 or 1
    int k = cur_data[attribute_count_total]; // class

    // int mask = cur_leaf_counter[attribute_count_total * 2 + ij];
    int n_ijk_idx = (k + 2) * attribute_count_total * 2 + ij;

    // atomicAdd(&cur_leaf_counter[ij], mask); // row 0
    // atomicAdd(&cur_leaf_counter[n_ijk_idx], mask);
    atomicAdd(&cur_leaf_counter[ij], cur_weight); // row 0
    atomicAdd(&cur_leaf_counter[n_ijk_idx], cur_weight);
}

__global__ void compute_information_gain(
        int *leaf_counters,
        int* is_leaf_active,
        int *tree_status,
        int *leaf_class,
        float *info_gain_vals,
        int *attribute_val_arr,
        int attribute_count_per_tree,
        int attribute_count_total,
        int leaf_count_per_tree,
        int class_count,
        int leaf_counter_size) {
    // each leaf_counter is mapped to one block in the 1D grid
    // one thread uses one whole column per leaf counter
    // each block needs as many threads as twice number of the (binary) attributes

    // output: a vector with the attributes information gain values for all leaves in each of the trees
    // gridDim: dim3(TREE_COUNT, INSTANCE_COUNT_PER_TREE)
    // blockDim: attributes_per_tree * 2 (equal to the length of a info_gain_vals per leaf)

    int block_id = blockIdx.y + blockIdx.x * gridDim.y;

    int thread_pos = threadIdx.x + block_id * blockDim.x;
    if (thread_pos >= gridDim.x * gridDim.y * blockDim.x) {
        return;
    }

    int tree_idx = blockIdx.x;

    int cur_tree_status = tree_status[tree_idx];
    if (cur_tree_status == 0 || cur_tree_status == 2) {
        return;
    }

    int leaf_id = blockIdx.y;

    int info_gain_per_len = attribute_count_per_tree * 2;
    int cur_tree_info_gain_start_pos = tree_idx * leaf_count_per_tree * info_gain_per_len;
    int cur_leaf_info_gain_start_pos = cur_tree_info_gain_start_pos + leaf_id *
        info_gain_per_len;
    float *cur_info_gain_vals = info_gain_vals + cur_leaf_info_gain_start_pos;

    if (is_leaf_active[tree_idx * leaf_count_per_tree + leaf_id] != 1) {
        cur_info_gain_vals[threadIdx.x] = FLT_MAX;
        return;
    }

    int leaf_counter_row_len = attribute_count_total * 2;

    int cur_tree_counters_start_pos = tree_idx * leaf_count_per_tree * leaf_counter_size;
    int cur_leaf_counter_start_pos = cur_tree_counters_start_pos + leaf_id * leaf_counter_size;
    int *cur_leaf_counter = leaf_counters + cur_leaf_counter_start_pos;

    int *cur_attribute_val_arr = attribute_val_arr + tree_idx * attribute_count_per_tree;


    int col_idx = cur_attribute_val_arr[threadIdx.x >> 1] * 2 + (threadIdx.x & 1);

    int a_ij = cur_leaf_counter[col_idx];
    int mask = cur_leaf_counter[leaf_counter_row_len + col_idx];

    // sum up a column
    float sum = 0.0;

    for (int i = 0; i < class_count; i++) {
        int a_ijk = cur_leaf_counter[col_idx + (2 + i) * leaf_counter_row_len];

        // 0/0 = inf
        float param = (float) a_ijk / (a_ij * mask);
        asm("max.f32 %0, %1, %2;" : "=f"(param) : "f"(param), "f"((float) 0.0));

        // log2(0) = -inf
        float log_param = log2f((float) param);
        asm("max.f32 %0, %1, %2;" : "=f"(log_param) : "f"(-log_param), "f"((float) 0.0));

        sum += param * log_param;
    }

    cur_info_gain_vals[threadIdx.x] = sum;

    __syncthreads();

    float i_00 = 0.0, i_01 = 0.0;
    int i_idx = 0;

    if ((threadIdx.x & 1) == 0) {
        i_00 = cur_info_gain_vals[threadIdx.x];
        i_01 = cur_info_gain_vals[threadIdx.x + 1];
        i_idx = (threadIdx.x >> 1);
    }

    __syncthreads();

    if ((threadIdx.x & 1) == 0) {
        cur_info_gain_vals[i_idx] = i_00 + i_01;
    }

    if (threadIdx.x != 0) {
        return;
    }

    int majority_class_code = 0;
    int majority_class_count = 0;

    for (int k = 0; k < class_count; k++) {
        int a_k = cur_leaf_counter[threadIdx.x + (2 + k) * leaf_counter_row_len]
                + cur_leaf_counter[threadIdx.x + 1 + (2 + k) * leaf_counter_row_len];

        if (a_k > majority_class_count) {
            majority_class_count = a_k;
            majority_class_code = k;
        }
    }

    int *cur_leaf_class = leaf_class + tree_idx * leaf_count_per_tree;
    cur_leaf_class[leaf_id] = majority_class_code;
}

// hoeffding bound
// providing an upper bound on the probability that the sum of a sample of independent random
// variables deviates from its expected value
//
// range: range of the random variable
// confidence: desired probability of the estimate not being within the expected value
// n: the number of examples collected at the node
__device__ float compute_hoeffding_bound(float range, float confidence, float n) {
    float result = sqrt(((range * range) * log(1.0 / confidence)) / (2.0 * n));
    // printf("=========> range: %f, confidence: %f, n: %f, result: %f\n", range, confidence, n, result);
    return result;
}

__global__ void compute_node_split_decisions(
        float* info_gain_vals,
        int* is_leaf_active,
        int* tree_status,
        int* attribute_val_arr,
        int* attribute_idx_arr,
        int* node_split_decisions,
        int attribute_count_per_tree,
        float r,
        float delta,
        int leaf_count_per_tree,
        int* samples_seen_count) {
    // <<<TREE_COUNT, LEAF_COUNT_PER_TREE>>>
    // note: different from paper by using one thread per leaf
    // output: an array of decisions
    //         - the most significant bit denotes whether a leaf needs to be split
    //         - the rest bits denote the attribute id to split on

    int thread_pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_pos >= gridDim.x * blockDim.x) {
        return;
    }

    int tree_idx = blockIdx.x;
    int cur_tree_status = tree_status[tree_idx];
    if (cur_tree_status == 0 || cur_tree_status == 2) {
        return;
    }

    int leaf_idx = threadIdx.x;
    if (is_leaf_active[tree_idx * leaf_count_per_tree + leaf_idx] != 1) {
        return;
    }

    int *cur_attribute_val_arr = attribute_val_arr + tree_idx * attribute_count_per_tree;

    int cur_tree_attr_idx_start_pos = tree_idx * leaf_count_per_tree * attribute_count_per_tree;
    int cur_leaf_attr_idx_start_pos = cur_tree_attr_idx_start_pos + leaf_idx *
        attribute_count_per_tree;
    int *cur_attribute_idx_arr = attribute_idx_arr + cur_leaf_attr_idx_start_pos;

    int cur_tree_info_gain_start_pos = tree_idx * leaf_count_per_tree
        * attribute_count_per_tree * 2;
    int cur_leaf_info_gain_start_pos = cur_tree_info_gain_start_pos + leaf_idx *
        attribute_count_per_tree * 2;
    float *cur_info_gain_vals = info_gain_vals + cur_leaf_info_gain_start_pos;

    thrust::sort_by_key(thrust::seq,
            cur_info_gain_vals,
            cur_info_gain_vals + attribute_count_per_tree,
            cur_attribute_idx_arr);


    // iteration is faster than thrust parallel sort
    // int first_best_idx = 0;
    // float first_best_val = FLT_MAX;
    // float second_best_val = FLT_MAX;

    // for (int i = 0; i < attribute_count_per_tree; i++) {
    //     if (first_best_val - cur_info_gain_vals[i] > EPS) {
    //         second_best_val = first_best_val;
    //         first_best_val = cur_info_gain_vals[i];

    //         first_best_idx = cur_attribute_val_arr[i];

    //     } else if (second_best_val - cur_info_gain_vals[i] > EPS) {
    //         second_best_val = cur_info_gain_vals[i];
    //     }
    // }

    float first_best_val = cur_info_gain_vals[0];
    float second_best_val = cur_info_gain_vals[1];

    float hoeffding_bound = compute_hoeffding_bound(r, delta, samples_seen_count[thread_pos]);

    int decision = 0;
    if (second_best_val - first_best_val - hoeffding_bound > EPS) {
        // split on the best attribute
        decision |= (1 << 31);
        // decision |= first_best_idx;
        decision |= cur_attribute_val_arr[cur_attribute_idx_arr[0]];
    }

    int* cur_node_split_decisions = node_split_decisions + tree_idx * leaf_count_per_tree;
    cur_node_split_decisions[leaf_idx] = decision;
}

__global__ void node_split(
        int *decision_trees,
        int *tree_status,
        int *node_split_decisions,
        int *leaf_counters,
        int *leaf_class,
        int *leaf_back,
        int *attribute_val_arr,
        int *samples_seen_count,
        int *cur_node_count_per_tree,
        int *cur_leaf_count_per_tree,
        int counter_size_per_leaf,
        int max_node_count_per_tree,
        int max_leaf_count_per_tree,
        int attribute_count_per_tree,
        int attribute_count_total,
        int class_count) {
    // <<<1, TREE_COUNT>>>
    // only launch one thread for each tree
    // to append new leaves at the end of the decision_tree array sequentially

    if (threadIdx.x >= blockDim.x) {
        return;
    }

    int tree_idx = threadIdx.x;
    int cur_tree_status = tree_status[tree_idx];

    if (cur_tree_status == 0 || cur_tree_status == 2) {
        // tree is either inactive or an inactive background tree
        return;
    }

    int cur_node_count = cur_node_count_per_tree[tree_idx];
    int cur_leaf_count = cur_leaf_count_per_tree[tree_idx];

    int *cur_decision_tree = decision_trees + tree_idx * max_node_count_per_tree;

    int *cur_node_split_decisions = node_split_decisions + tree_idx *
        max_leaf_count_per_tree;

    int *cur_tree_leaf_counters = leaf_counters +
        tree_idx * max_leaf_count_per_tree * counter_size_per_leaf;

    int *cur_leaf_back = leaf_back + tree_idx * max_leaf_count_per_tree;
    int *cur_leaf_class = leaf_class + tree_idx * max_leaf_count_per_tree;

    int *cur_attribute_val_arr = attribute_val_arr + tree_idx * attribute_count_per_tree;

    for (int leaf_idx = 0; leaf_idx < max_leaf_count_per_tree; leaf_idx++) {
        unsigned int decision = cur_node_split_decisions[leaf_idx];

        if (!IS_BIT_SET(decision, 31)) {
            continue;
        }

        cur_node_split_decisions[leaf_idx] = 0;

        int *cur_leaf_counter = cur_tree_leaf_counters + leaf_idx * counter_size_per_leaf;

        if (cur_node_count == max_node_count_per_tree) {
            // tree is full
            return;
        }

        int attribute_id = (decision & ~(1 << 31));
        int cur_leaf_pos_in_tree = cur_leaf_back[leaf_idx];
        int cur_leaf_val = cur_decision_tree[cur_leaf_pos_in_tree];

        int old_leaf_id = (cur_leaf_val & ~(1 << 31));
        int new_leaf_id = cur_leaf_count;

        int *cur_samples_seen_count = samples_seen_count + tree_idx * max_leaf_count_per_tree;

        cur_samples_seen_count[old_leaf_id] = 0;
        cur_samples_seen_count[new_leaf_id] = 0;

        int left_leaf_pos = get_left(cur_leaf_pos_in_tree);
        int right_leaf_pos = get_right(cur_leaf_pos_in_tree);

        if (left_leaf_pos >= max_node_count_per_tree
                || right_leaf_pos >= max_node_count_per_tree) {
            continue;
        }

        cur_decision_tree[cur_leaf_pos_in_tree] = attribute_id;
        // cur_decision_tree[cur_leaf_pos_in_tree] = cur_attribute_val_arr[attribute_id];

        cur_decision_tree[left_leaf_pos] = cur_leaf_val;
        cur_decision_tree[right_leaf_pos] = (1 << 31) | new_leaf_id;

        cur_leaf_back[old_leaf_id] = left_leaf_pos;
        cur_leaf_back[new_leaf_id] = right_leaf_pos;


        int left_max_class_code = 0;
        int left_max_count = cur_leaf_counter[attribute_count_total * 2 * 2
            + attribute_id * 2];

        int right_max_class_code = 0;
        int right_max_count = cur_leaf_counter[attribute_count_total * 2 * 2
            + attribute_id * 2 + 1];

        for (int k = 1; k < class_count; k++) {
            // left
            int cur_left_class_count = cur_leaf_counter[attribute_count_total * 2 * (k + 2) +
                attribute_id * 2];
            if (cur_left_class_count > left_max_count) {
                left_max_count = cur_left_class_count;
                left_max_class_code = k;
            }

            // right
            int cur_right_class_count = cur_leaf_counter[attribute_count_total * 2 * (k + 2) +
                attribute_id * 2 + 1];
            if (cur_right_class_count > right_max_count) {
                right_max_count = cur_right_class_count;
                right_max_class_code = k;
            }
        }

        cur_leaf_class[old_leaf_id] = left_max_class_code;
        cur_leaf_class[new_leaf_id] = right_max_class_code;


        // reset current leaf_counter and add copy mask to a new leaf counter
        int *new_leaf_counter = cur_tree_leaf_counters + cur_leaf_count * counter_size_per_leaf;

        for (int k = 0; k < class_count + 2; k++) {
            int *cur_leaf_counter_row = cur_leaf_counter + attribute_count_total * 2 * k;
            int *new_leaf_counter_row = new_leaf_counter + attribute_count_total * 2 * k;

            if (k == 1) {
                for (int ij = 0; ij < attribute_count_total * 2; ij++) {
                    if (ij == attribute_id * 2 || ij == attribute_id * 2 + 1) {
                        cur_leaf_counter_row[ij] = 0;
                    }

                    new_leaf_counter_row[ij] = cur_leaf_counter_row[ij];
                }

            } else {
                for (int ij = 0; ij < attribute_count_total * 2; ij++) {
                    cur_leaf_counter_row[ij] = 0;
                    new_leaf_counter_row[ij] = 0;
                }
            }
        }

        cur_node_count += 2;
        cur_leaf_count += 1;
    }

    cur_node_count_per_tree[tree_idx] = cur_node_count;
    cur_leaf_count_per_tree[tree_idx] = cur_leaf_count;
}

int main(int argc, char *argv[]) {

    int TREE_COUNT = 1;
    int TREE_DEPTH_PARAM = -1;
    int INSTANCE_COUNT_PER_TREE = 200;
    int SAMPLE_FREQUENCY = 1000;


    float n_min = 50; // hoeffding bound parameter, grace_period
    double kappa_threshold = 0.1;
    int edit_distance_threshold = 50;

    string data_path = "data/covtype";
    string data_file_name = "covtype_binary_attributes.csv";

    int opt;
    while ((opt = getopt(argc, argv, "t:i:p:n:s:d:g:k:e:r")) != -1) {
        switch (opt) {
            case 't':
                TREE_COUNT = atoi(optarg);
                break;
            case 'i':
                INSTANCE_COUNT_PER_TREE = atoi(optarg);
                break;
            case 'p':
                data_path = optarg;
                break;
            case 'n':
                data_file_name = optarg;
                break;
            case 's':
                SAMPLE_FREQUENCY = atoi(optarg);
                break;
            case 'd':
                TREE_DEPTH_PARAM = atoi(optarg);
                break;
            case 'g':
                n_min = atoi(optarg);
                break;
            case 'k':
                kappa_threshold = atof(optarg);
                break;
            case 'e':
                edit_distance_threshold = atoi(optarg);
                break;
            case 'r':
                // Use a different seed value for each run
                srand(time(NULL));
                break;
        }
    }

    int FOREGROUND_TREE_COUNT = TREE_COUNT;
    int GROWING_TREE_COUNT = TREE_COUNT * 2;
    TREE_COUNT *= 3;

    ofstream log_file;
    log_file.open("log_file.txt");

    log_file << "TREE_COUNT = " << TREE_COUNT << endl
        << "GROWING_TREE_COUNT = " << GROWING_TREE_COUNT << endl
        << "INSTANCE_COUNT_PER_TREE = " << INSTANCE_COUNT_PER_TREE << endl;

    log_file << "edit_distance_threshold: " << edit_distance_threshold << endl
        << "kappa_threshold: " << endl;

    string output_path = data_path + "/result_gpu.csv";
    ofstream output_file;
    output_file.open(output_path);

    log_file << endl;
    if (output_file.fail()) {
        log_file << "Error opening output file at " << output_path << endl;
        return 1;
    } else {
        log_file << "Writing output to " << output_path << endl;
    }


    // read data file
    string attribute_file_path = data_path + "/attributes.txt";
    ifstream attribute_file(attribute_file_path);

    log_file << endl;
    if (attribute_file) {
        log_file << "Reading data file from " << attribute_file_path << " succeeded." << endl;
    } else {
        log_file << "Error reading file from " << attribute_file_path << endl;
        return 1;
    }

    // prepare attributes
    string line;
    getline(attribute_file, line);

    const int ATTRIBUTE_COUNT_TOTAL = split(line, ",").size() - 1;
    const int ATTRIBUTE_COUNT_PER_TREE = (int) sqrt(ATTRIBUTE_COUNT_TOTAL) + 1;

    const int TREE_DEPTH =
        TREE_DEPTH_PARAM == -1 ? (int) sqrt(ATTRIBUTE_COUNT_TOTAL) + 1 : TREE_DEPTH_PARAM;

    log_file << "ATTRIBUTE_COUNT_TOTAL = " << ATTRIBUTE_COUNT_TOTAL << endl;
    log_file << "ATTRIBUTE_COUNT_PER_TREE = " << ATTRIBUTE_COUNT_PER_TREE << endl;
    log_file << "TREE_DEPTH = " << TREE_DEPTH << endl;

    NODE_COUNT_PER_TREE = (1 << TREE_DEPTH) - 1;
    LEAF_COUNT_PER_TREE = (1 << (TREE_DEPTH - 1));

    log_file << "NODE_COUNT_PER_TREE = " << NODE_COUNT_PER_TREE << endl;
    log_file << "LEAF_COUNT_PER_TREE = " << LEAF_COUNT_PER_TREE << endl;


    // read class/label file
    string class_path = data_path + "/labels.txt";
    ifstream class_file(class_path);

    log_file << endl;
    if (class_file) {
        log_file << "Reading class file from " << class_path << " succeeded." << endl;
    } else {
        log_file << "Error reading class file from " << class_path << endl;
    }

    string class_line;

    // init mapping between class and code
    map<string, int> class_code_map;
    map<int, string> code_class_map;

    vector<string> class_arr = split(class_line, " ");
    string code_str, class_str;

    int line_count = 0;
    while (class_file >> class_str) {
        int class_code = line_count;
        class_code_map[class_str] = class_code;
        code_class_map[class_code] = class_str;
        line_count++;
    }
    const int CLASS_COUNT = line_count;
    log_file << "CLASS_COUNT = " << CLASS_COUNT << endl;

    // hoeffding bound parameters
    float delta = 0.05; // pow((float) 10.0, -7);
    float r = log2(CLASS_COUNT); // range of merit = log2(num_of_classes)

    log_file << endl
        << "hoeffding bound parameters: " << endl
        << "n_min = " << n_min << endl
        << "delta = " << delta << endl
        << "r     = " << r     << endl;


    // init decision tree
    log_file << "\nAllocating memory on host..." << endl;
    // void *allocated = malloc(NODE_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    void *allocated = calloc(NODE_COUNT_PER_TREE * TREE_COUNT, sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for decision trees failed" << endl;
        return 1;
    }
    int *h_decision_trees = (int*) allocated;

    int *d_decision_trees;
    if (!allocate_memory_on_device(&d_decision_trees, "decision_trees", NODE_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }

    // CPU tree pool allocations
    int CPU_TREE_POOL_SIZE = TREE_COUNT * 10;
    int cur_tree_pool_size = FOREGROUND_TREE_COUNT;

    allocated = malloc(CPU_TREE_POOL_SIZE * NODE_COUNT_PER_TREE * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for cpu tree pool failed" << endl;
        return 1;
    }
    int* cpu_decision_trees = (int*) allocated;


    log_file << "Init: set root as leaf for each tree in the forest..." << endl;

    for (int i = 0; i < TREE_COUNT; i++) {
        int *cur_decision_tree = h_decision_trees + i * NODE_COUNT_PER_TREE;
        cur_decision_tree[0] = (1 << 31);

        for (int j = 1; j < NODE_COUNT_PER_TREE; j++) {
            cur_decision_tree[j] = -1;
        }
    }

    int* forest_idx_to_tree_id = (int*) malloc(TREE_COUNT * sizeof(int));
    memset(forest_idx_to_tree_id, -1, TREE_COUNT * sizeof(int));

    int* tree_id_to_forest_idx = (int*) malloc(CPU_TREE_POOL_SIZE * sizeof(int));
    memset(tree_id_to_forest_idx, -1, CPU_TREE_POOL_SIZE * sizeof(int));

    for (int i = 0; i < FOREGROUND_TREE_COUNT; i++) {
        forest_idx_to_tree_id[i] = i;
        tree_id_to_forest_idx[i] = i;
    }

    gpuErrchk(cudaMemcpy(d_decision_trees, h_decision_trees, NODE_COUNT_PER_TREE * TREE_COUNT
                * sizeof(int), cudaMemcpyHostToDevice));


    allocated = malloc(LEAF_COUNT_PER_TREE * CPU_TREE_POOL_SIZE * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for cpu_leaf_class failed" << endl;
        return 1;
    }
    int *cpu_leaf_class = (int*) allocated; // stores the class for a given leaf

    allocated = malloc(LEAF_COUNT_PER_TREE *  CPU_TREE_POOL_SIZE * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for cpu_leaf_back failed" << endl;
        return 1;
    }
    int *cpu_leaf_back = (int*) allocated; // reverse pointer to map a leaf id to an offset in the tree array


    allocated = malloc(LEAF_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for leaf_class failed" << endl;
        return 1;
    }
    int *h_leaf_class = (int*) allocated; // stores the class for a given leaf

    allocated = malloc(LEAF_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for leaf_back failed" << endl;
        return 1;
    }
    int *h_leaf_back = (int*) allocated; // reverse pointer to map a leaf id to an offset in the tree array


    // the offsets of leaves reached from tree traversal
    int *d_reached_leaf_ids;
    if (!allocate_memory_on_device(&d_reached_leaf_ids, "leaf_ids",
                INSTANCE_COUNT_PER_TREE * GROWING_TREE_COUNT)) {
        return 1;
    }

    int *d_is_leaf_active;
    if (!allocate_memory_on_device(&d_is_leaf_active, "is_leaf_active",
                LEAF_COUNT_PER_TREE * GROWING_TREE_COUNT)) {
        return 1;
    }

    int *d_leaf_class;
    if (!allocate_memory_on_device(&d_leaf_class, "leaf_class", LEAF_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }

    int *d_leaf_back;
    if (!allocate_memory_on_device(&d_leaf_back, "leaf_back", LEAF_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }

    int LEAF_COUNTER_SIZE = ATTRIBUTE_COUNT_TOTAL * 2 * (CLASS_COUNT + 2);
    LEAF_COUNTERS_SIZE_PER_TREE = LEAF_COUNT_PER_TREE * LEAF_COUNTER_SIZE;
    int ALL_LEAF_COUNTERS_SIZE = GROWING_TREE_COUNT * LEAF_COUNTERS_SIZE_PER_TREE;

    int* h_leaf_counters = (int*) calloc(ALL_LEAF_COUNTERS_SIZE, sizeof(int));

    long cpu_leaf_counters_size = (long) LEAF_COUNTERS_SIZE_PER_TREE
        * CPU_TREE_POOL_SIZE;
    int* cpu_leaf_counters = (int*) malloc(cpu_leaf_counters_size * sizeof(int));

    // init mask row
    for (int tree_idx = 0; tree_idx < GROWING_TREE_COUNT; tree_idx++) {
        int *cur_tree_leaf_counters = h_leaf_counters + tree_idx * LEAF_COUNT_PER_TREE *
            LEAF_COUNTER_SIZE;
        for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
            int *cur_leaf_counter = cur_tree_leaf_counters + leaf_idx * LEAF_COUNTER_SIZE;
            int *cur_leaf_counter_mask_row = cur_leaf_counter + ATTRIBUTE_COUNT_TOTAL * 2;

            for (int k = 0; k < ATTRIBUTE_COUNT_TOTAL * 2; k++) {
                cur_leaf_counter_mask_row[k] = 1;
            }
        }
    }

    int *d_leaf_counters;
    if (!allocate_memory_on_device(&d_leaf_counters, "leaf_counters", ALL_LEAF_COUNTERS_SIZE)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_leaf_counters, h_leaf_counters, ALL_LEAF_COUNTERS_SIZE * sizeof(int),
                cudaMemcpyHostToDevice));

    // TODO: h_info_gain_vals for testing only
    int info_gain_vals_len = GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE * ATTRIBUTE_COUNT_PER_TREE * 2;
    float *h_info_gain_vals = (float*) malloc(info_gain_vals_len * sizeof(float));

    float *d_info_gain_vals;
    if (!allocate_memory_on_device(&d_info_gain_vals, "info_gain_vals", info_gain_vals_len)) {
        return 1;
    }


    // actual selected attributes for each tree for counter_increase kernel
    int *h_attribute_val_arr;
    int *d_attribute_val_arr;
    int attribute_val_arr_len = GROWING_TREE_COUNT * ATTRIBUTE_COUNT_PER_TREE;

    allocated = malloc(attribute_val_arr_len * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for h_attribute_val_arr failed" << endl;
    }
    h_attribute_val_arr = (int*) allocated;


    if (!allocate_memory_on_device(&d_attribute_val_arr, "attribute_val_arr",
                attribute_val_arr_len)) {
        return 1;
    }

    // allocate memory for attribute indices on host for computing information gain
    int *h_attribute_idx_arr;
    int *d_attribute_idx_arr;
    int attribute_idx_arr_len = GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE * ATTRIBUTE_COUNT_PER_TREE;

    allocated = malloc(attribute_idx_arr_len * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for h_attribute_idx_arr failed" << endl;
        return 1;
    }
    h_attribute_idx_arr = (int*) allocated;

    if (!allocate_memory_on_device(&d_attribute_idx_arr, "attribute_idx_arr",
                attribute_idx_arr_len)) {
        return 1;
    }

    for (int tree_idx = 0; tree_idx < GROWING_TREE_COUNT; tree_idx++) {
        int *cur_tree_attribute_idx_arr = h_attribute_idx_arr + tree_idx * LEAF_COUNT_PER_TREE
            * ATTRIBUTE_COUNT_PER_TREE;

        for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
            int *cur_attribute_idx_arr = cur_tree_attribute_idx_arr + leaf_idx *
                ATTRIBUTE_COUNT_PER_TREE;

            for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
                cur_attribute_idx_arr[i] = i;
            }
        }
    }

    // TODO same attribute_idx_arr for foreground and its background tree

    // allocate memory for node_split_decisions
    // unsigned int *h_node_split_decisions;
    int *d_node_split_decisions;
    int node_split_decisions_len = LEAF_COUNT_PER_TREE * GROWING_TREE_COUNT;

    // allocated = malloc(node_split_decisions_len * sizeof(unsigned int));
    // if (allocated == NULL) {
    //     log_file << "host error: memory allocation for h_node_split_decisions failed" << endl;
    //     return 1;
    // }
    // h_node_split_decisions = (unsigned int*) allocated;

    if (!allocate_memory_on_device(&d_node_split_decisions, "node_split_decisions",
                node_split_decisions_len)) {
        return 1;
    }

    int samples_seen_count_len = GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE;
    int *h_samples_seen_count = (int*) calloc(samples_seen_count_len, sizeof(int));
    int *cpu_samples_seen_count = (int*) calloc(LEAF_COUNT_PER_TREE * CPU_TREE_POOL_SIZE,
            sizeof(int));
    int *d_samples_seen_count;
    if (!allocate_memory_on_device(&d_samples_seen_count, "samples_seen_count",
                samples_seen_count_len)) {
        return 1;
    }

    int h_cur_node_count_per_tree[GROWING_TREE_COUNT];
    int cpu_cur_node_count_per_tree[CPU_TREE_POOL_SIZE];
    int *d_cur_node_count_per_tree;

    fill_n(h_cur_node_count_per_tree, GROWING_TREE_COUNT, 1);

    if (!allocate_memory_on_device(&d_cur_node_count_per_tree, "cur_node_count_per_tree",
                GROWING_TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_cur_node_count_per_tree, h_cur_node_count_per_tree,
                GROWING_TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));

    int h_cur_leaf_count_per_tree[GROWING_TREE_COUNT];
    int cpu_cur_leaf_count_per_tree[CPU_TREE_POOL_SIZE];
    int *d_cur_leaf_count_per_tree;

    fill_n(h_cur_leaf_count_per_tree, GROWING_TREE_COUNT, 1);

    if (!allocate_memory_on_device(&d_cur_leaf_count_per_tree,
                "leaf_count_per_tree", GROWING_TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_cur_leaf_count_per_tree, h_cur_leaf_count_per_tree,
                 GROWING_TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));


    int forest_vote_len = INSTANCE_COUNT_PER_TREE * CLASS_COUNT;
    int *d_forest_vote;
    if (!allocate_memory_on_device(&d_forest_vote, "forest_vote", forest_vote_len)) {
        return 1;
    }

    int h_forest_vote_idx_arr[forest_vote_len];
    for (int i = 0; i < INSTANCE_COUNT_PER_TREE; i++) {
        for (int j = 0; j < CLASS_COUNT; j++) {
            h_forest_vote_idx_arr[i * CLASS_COUNT + j] = j;
        }
    }
    int *d_forest_vote_idx_arr;
    if (!allocate_memory_on_device(&d_forest_vote_idx_arr, "forest_vote_idx_arr",
                forest_vote_len)) {
        return 1;
    }

    int *d_weights;
    if (!allocate_memory_on_device(&d_weights, "weights", GROWING_TREE_COUNT * INSTANCE_COUNT_PER_TREE)) {
        return 1;
    }

    // one warning and drift detector per tree to monitor accuracy
    // initialized with the default construct where delta=0.001
    ADWIN* warning_detectors[FOREGROUND_TREE_COUNT];
    ADWIN* drift_detectors[FOREGROUND_TREE_COUNT];

    for (int i = 0; i < FOREGROUND_TREE_COUNT; i++) {
        warning_detectors[i] = new ADWIN((double) 0.001);
        drift_detectors[i] = new ADWIN((double) 0.00001);
    }

    int tree_error_count_len = TREE_COUNT;
    int* h_tree_error_count = (int*) calloc(tree_error_count_len, sizeof(int));
    int* d_tree_error_count;
    if (!allocate_memory_on_device(&d_tree_error_count, "tree_error_count", tree_error_count_len)) {
        return 1;
    }

    int* d_drift_tree_idx_arr;
    if (!allocate_memory_on_device(&d_drift_tree_idx_arr, "drift_tree_idx_arr",
                GROWING_TREE_COUNT)) {
        return 1;
    }

    int* d_warning_tree_idx_arr;
    if (!allocate_memory_on_device(&d_warning_tree_idx_arr, "warning_tree_idx_arr",
                GROWING_TREE_COUNT)) {
        return 1;
    }

    // pointer to the start of the background decision trees
    int *h_background_trees = h_decision_trees + FOREGROUND_TREE_COUNT * NODE_COUNT_PER_TREE;


    // for swapping background trees when drift is detected
    state_graph* state_transition_graph = new state_graph(CPU_TREE_POOL_SIZE);
    LRU_state* state_queue = new LRU_state(100, edit_distance_threshold);

    // TODO
    // 0: inactive, 1: active, 2: must be inactive
    // add initial state
    vector<char> cur_state(CPU_TREE_POOL_SIZE);

    for (int i = 0; i < FOREGROUND_TREE_COUNT; i++) {
        cur_state[i] = '1';
    }

    for (int i = FOREGROUND_TREE_COUNT; i < CPU_TREE_POOL_SIZE; i++) {
        cur_state[i] = '0';
    }


    // TODO
    // 0: inactive, 1: active, 2: inactive bg_tree, 3: active bg_tree
    int h_tree_active_status[TREE_COUNT];
    int *d_tree_active_status;
    if (!allocate_memory_on_device(&d_tree_active_status, "d_tree_active_status", TREE_COUNT)) {
        return 1;
    }

    for (int i = 0; i < FOREGROUND_TREE_COUNT; i++) {
        h_tree_active_status[i] = 1;
    }

    for (int i = FOREGROUND_TREE_COUNT; i < (FOREGROUND_TREE_COUNT << 1); i++) {
        h_tree_active_status[i] = 2;
    }

    for (int i = (FOREGROUND_TREE_COUNT << 1); i < TREE_COUNT; i++) {
        h_tree_active_status[i] = 4;
    }

    queue<int> next_empty_forest_idx;
    for (int i = GROWING_TREE_COUNT; i < TREE_COUNT; i++) {
        next_empty_forest_idx.push(i);
    }

    vector<candidate_t> forest_candidate_vec; // candidates in forest

    gpuErrchk(cudaMemcpy(d_tree_active_status, h_tree_active_status,
                TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));

#if DEBUG

    cout << "initial cur_state: ";
    for (int i = 0; i < cur_state.size(); i++) {
        cout << cur_state[i];
    }
    cout << endl;

    cout << "tree active status: ";
    for (int i = 0; i < TREE_COUNT; i++) {
        cout << h_tree_active_status[i] << " ";
    }
    cout << endl;;

#endif


    // for calculating kappa measurements
    int confusion_matrix_size = CLASS_COUNT * CLASS_COUNT;
    int *h_confusion_matrix = (int*) malloc(confusion_matrix_size * sizeof(int));
    int *d_confusion_matrix;
    if (!allocate_memory_on_device(&d_confusion_matrix, "d_confusion_matrix",
                confusion_matrix_size)) {
        return 1;
    }

    int *h_tree_confusion_matrix = (int*) malloc(TREE_COUNT * confusion_matrix_size * sizeof(int));
    int *d_tree_confusion_matrix;
    if (!allocate_memory_on_device(&d_tree_confusion_matrix, "d_tree_confusion_matrix",
                TREE_COUNT * confusion_matrix_size)) {
        return 1;
    }

    int *h_tree_accuracy = (int*) malloc(TREE_COUNT * sizeof(int));
    int *d_tree_accuracy;


    vector<tree_t> h_forest;
    for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
        tree_t cur_tree = {
            h_decision_trees + tree_idx * NODE_COUNT_PER_TREE,
            h_leaf_class + tree_idx * LEAF_COUNT_PER_TREE,
            h_leaf_back + tree_idx * LEAF_COUNT_PER_TREE,
            h_leaf_counters + tree_idx * LEAF_COUNTERS_SIZE_PER_TREE,
            h_cur_node_count_per_tree + tree_idx,
            h_cur_leaf_count_per_tree + tree_idx,
            h_samples_seen_count + tree_idx * LEAF_COUNT_PER_TREE,
        };
        h_forest.push_back(cur_tree);
    }


    vector<tree_t> cpu_forest;
    for (int tree_id = 0; tree_id < CPU_TREE_POOL_SIZE; tree_id++) {
        tree_t cur_tree = {
            cpu_decision_trees + tree_id * NODE_COUNT_PER_TREE,
            cpu_leaf_class + tree_id * LEAF_COUNT_PER_TREE,
            cpu_leaf_back + tree_id * LEAF_COUNT_PER_TREE,
            cpu_leaf_counters + tree_id * LEAF_COUNTERS_SIZE_PER_TREE,
            cpu_cur_node_count_per_tree + tree_id,
            cpu_cur_leaf_count_per_tree + tree_id,
            cpu_samples_seen_count + tree_id * LEAF_COUNT_PER_TREE,
        };
        cpu_forest.push_back(cur_tree);
    }


    log_file << "\nInitializing training data arrays..." << endl;

    int data_len = INSTANCE_COUNT_PER_TREE * (ATTRIBUTE_COUNT_TOTAL + 1);
    int *h_data = (int*) malloc(data_len * sizeof(int));

    int *d_data;
    if (!allocate_memory_on_device(&d_data, "data", data_len)) {
        return 1;
    }

    int *d_class_count_arr;
    if (!allocate_memory_on_device(&d_class_count_arr, "class_count_arr", CLASS_COUNT)) {
        return 1;
    }

    // read data file
    string csv_path = data_path + "/" + data_file_name;
    ifstream data_file(csv_path);

    log_file << endl;
    if (data_file) {
        log_file << "Reading data file from " << csv_path << " succeeded." << endl;
    } else {
        log_file << "Error reading file from " << csv_path << endl;
        return 1;
    }

    vector<string> raw_data_row;

    int block_count;
    int thread_count;

    log_file << endl << "=====Training Start=====" << endl;

    int h_correct_counter = 0;
    int *d_correct_counter;
    gpuErrchk(cudaMalloc((void **) &d_correct_counter, sizeof(int)));

    curandState *d_state;
    cudaMalloc(&d_state, GROWING_TREE_COUNT * INSTANCE_COUNT_PER_TREE * sizeof(curandState));

    setup_kernel<<<GROWING_TREE_COUNT, INSTANCE_COUNT_PER_TREE>>>(d_state);
    cudaDeviceSynchronize();

    int leaf_counter_row_len = ATTRIBUTE_COUNT_TOTAL * 2;
    int iter_count = 1;

    int sample_count_iter = 0;
    int sample_count_total = 0;
    double window_accuracy = 0.0;
    double window_kappa = 0.0;

    // output_file << "#iteration,accuracy,mean_accuracy,kappa,mean_kappa" << endl;
    output_file << "#iteration,accuracy,kappa" << endl;

    bool eof = false;
    int matched_pattern = 0;

    while (!eof) {

        int h_data_idx = 0;
        int class_count_arr[CLASS_COUNT] = { 0 };

        for (int instance_idx = 0; instance_idx < INSTANCE_COUNT_PER_TREE; instance_idx++) {
            if (!getline(data_file, line)) {
                eof = true;
                break;
            }

            raw_data_row = split(line, ",");

            for (int i = 0; i < ATTRIBUTE_COUNT_TOTAL; i++) {
                int val = stoi(raw_data_row[i]);
                h_data[h_data_idx++] = val;
            }

            int cur_class_code = class_code_map[raw_data_row[ATTRIBUTE_COUNT_TOTAL]];
            h_data[h_data_idx] = cur_class_code;
            class_count_arr[cur_class_code]++;

            h_data_idx++;
        }

        if (eof) {
            break;
        }

        log_file << endl << "=================iteration " << iter_count
            << "=================" << endl;

        int majority_class = 0;
        int majority_class_count = 0;

        for (int i = 0; i < CLASS_COUNT; i++) {
            if (majority_class_count < class_count_arr[i]) {
                majority_class_count = class_count_arr[i];
                majority_class = i;
            }
        }

        gpuErrchk(cudaMemcpy((void *) d_data, (void *) h_data, data_len
                    * sizeof(int), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy((void *) d_class_count_arr, (void *) class_count_arr, CLASS_COUNT
                    * sizeof(int), cudaMemcpyHostToDevice));

        log_file << "\nlaunching tree_traversal kernel..." << endl;

        block_count = TREE_COUNT;
        thread_count = INSTANCE_COUNT_PER_TREE;

        gpuErrchk(cudaMemset(d_correct_counter, 0, sizeof(int)));
        gpuErrchk(cudaMemset(d_tree_error_count, 0, tree_error_count_len * sizeof(int)));
        gpuErrchk(cudaMemset(d_confusion_matrix, 0, confusion_matrix_size * sizeof(int)));
        gpuErrchk(cudaMemset(d_tree_confusion_matrix, 0, TREE_COUNT * confusion_matrix_size * sizeof(int)));

        gpuErrchk(cudaMemset(d_is_leaf_active, 0, GROWING_TREE_COUNT
                    * LEAF_COUNT_PER_TREE * sizeof(int)));
        gpuErrchk(cudaMemset(d_forest_vote, 0, forest_vote_len * sizeof(int)));
        gpuErrchk(cudaMemcpy(d_forest_vote_idx_arr, h_forest_vote_idx_arr, forest_vote_len *
                    sizeof(int), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_tree_active_status, h_tree_active_status,
                    TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));

        log_file << "launching " << block_count * thread_count << " threads for tree_traversal" << endl;

        tree_traversal<<<block_count, thread_count>>>(
                d_decision_trees,
                d_tree_active_status,
                d_data,
                d_reached_leaf_ids,
                d_is_leaf_active,
                d_leaf_class,
                d_correct_counter,
                d_samples_seen_count,
                d_forest_vote,
                d_forest_vote_idx_arr,
                d_weights,
                d_tree_error_count,
                d_confusion_matrix,
                d_tree_confusion_matrix,
                d_class_count_arr,
                majority_class,
                NODE_COUNT_PER_TREE,
                LEAF_COUNT_PER_TREE,
                ATTRIBUTE_COUNT_TOTAL,
                CLASS_COUNT,
                confusion_matrix_size,
                d_state);


#if DEBUG

        gpuErrchk(cudaMemcpy(h_decision_trees, d_decision_trees, TREE_COUNT * NODE_COUNT_PER_TREE *
                    sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaMemcpy(h_leaf_class, d_leaf_class, TREE_COUNT * LEAF_COUNT_PER_TREE *
                    sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaMemcpy((void *) h_samples_seen_count, (void *) d_samples_seen_count,
                    samples_seen_count_len * sizeof(int), cudaMemcpyDeviceToHost));

#endif

        cudaDeviceSynchronize();
        log_file << "tree_traversal completed" << endl;

        gpuErrchk(cudaMemcpy(&h_correct_counter, d_correct_counter, sizeof(int),
                    cudaMemcpyDeviceToHost));

        log_file << "h_correct_counter: " << h_correct_counter << endl;

        double accuracy = (double) h_correct_counter / INSTANCE_COUNT_PER_TREE;
        window_accuracy = (sample_count_iter * window_accuracy + accuracy)
            / (sample_count_iter + 1);

        gpuErrchk(cudaMemcpy(h_confusion_matrix, d_confusion_matrix,
                    confusion_matrix_size * sizeof(int), cudaMemcpyDeviceToHost));

        double kappa = get_kappa(h_confusion_matrix, CLASS_COUNT, accuracy,
                INSTANCE_COUNT_PER_TREE);
        window_kappa = (sample_count_iter * window_kappa + kappa) / (sample_count_iter + 1);

        // log_file << "\n=================statistics" << endl
        //     << "accuracy: " << accuracy << endl
        //     << "kappa: " << kappa << endl;

        sample_count_iter++;;
        sample_count_total = sample_count_iter * INSTANCE_COUNT_PER_TREE; // avoid expensive mod

        if (sample_count_total >= SAMPLE_FREQUENCY) {
            output_file << iter_count * INSTANCE_COUNT_PER_TREE
                << "," << window_accuracy * 100
                << "," << window_kappa * 100 << endl;

            sample_count_iter = 0;
            window_accuracy = 0.0;
            window_kappa = 0.0;
        }


#if DEBUG

        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            log_file << "tree " << tree_idx << endl;
            int *cur_decision_tree = h_decision_trees + tree_idx * NODE_COUNT_PER_TREE;
            int *cur_leaf_class = h_leaf_class + tree_idx * LEAF_COUNT_PER_TREE;
            int *cur_samples_seen_count = h_samples_seen_count + tree_idx * LEAF_COUNT_PER_TREE;

            for (int i = 0; i < NODE_COUNT_PER_TREE; i++) {
                log_file << cur_decision_tree[i] << " ";
            }
            log_file << endl;

            for (int i = 0; i < LEAF_COUNT_PER_TREE; i++) {
                log_file << cur_leaf_class[i] << " ";
            }
            log_file << endl;

            log_file << "samples seen count: " << endl;
            for (int i = 0; i < LEAF_COUNT_PER_TREE; i++) {
                log_file << cur_samples_seen_count[i] << " ";
            }
            log_file << endl;
        }

#endif


        log_file << "\nlaunching counter_increase kernel..." << endl;

        counter_increase
            <<<dim3(GROWING_TREE_COUNT, INSTANCE_COUNT_PER_TREE),
            ATTRIBUTE_COUNT_TOTAL>>>(
                    d_leaf_counters,
                    d_tree_active_status,
                    d_reached_leaf_ids,
                    d_data,
                    d_weights,
                    CLASS_COUNT,
                    ATTRIBUTE_COUNT_TOTAL,
                    LEAF_COUNT_PER_TREE,
                    LEAF_COUNTER_SIZE);

        cudaDeviceSynchronize();
        log_file << "counter_increase completed" << endl;

#if DEBUG

        gpuErrchk(cudaMemcpy(h_leaf_counters, d_leaf_counters, ALL_LEAF_COUNTERS_SIZE
                    * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_cur_leaf_count_per_tree, d_cur_leaf_count_per_tree, TREE_COUNT
                    * sizeof(int), cudaMemcpyDeviceToHost));


        log_file << "counter_increase result: " << endl;
        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            log_file << "tree " << tree_idx << endl;

            log_file << "h_cur_leaf_count_per_tree is: " << h_cur_leaf_count_per_tree[tree_idx] << endl;
            int *cur_tree_leaf_counter = h_leaf_counters + tree_idx * LEAF_COUNT_PER_TREE
                * LEAF_COUNTER_SIZE;

            for (int leaf_idx = 0; leaf_idx < h_cur_leaf_count_per_tree[tree_idx]; leaf_idx++) {
                int *cur_leaf_counter = cur_tree_leaf_counter + leaf_idx * LEAF_COUNTER_SIZE;
                for (int k = 0; k < CLASS_COUNT + 2; k++) {
                    log_file << "row " << k << ": ";
                    for (int ij = 0; ij < leaf_counter_row_len; ij++) {
                        log_file << right << setw(8)
                            << cur_leaf_counter[k * leaf_counter_row_len + ij] << " ";
                    }
                    log_file << endl;
                }
            }
            log_file << endl;
        }

#endif

        log_file << "\nlanuching compute_information_gain kernel..." << endl;


        // select k random attributes for each tree
        // output_file << "\nAttributes selected per tree: " << endl;

        for (int tree_idx = 0; tree_idx < GROWING_TREE_COUNT; tree_idx++) {

            // select random attributes for active trees only
            if (h_tree_active_status[tree_idx] == 0 || h_tree_active_status[tree_idx] == 2) {
                continue;
            }

            // output_file << "tree " << tree_idx << endl;

            int *cur_attribute_val_arr = h_attribute_val_arr + tree_idx * ATTRIBUTE_COUNT_PER_TREE;
            select_k_attributes(cur_attribute_val_arr, ATTRIBUTE_COUNT_TOTAL,
                    ATTRIBUTE_COUNT_PER_TREE);

            // for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
            //     output_file << cur_attribute_val_arr[i] << " ";
            // }
            // output_file << endl;
        }

        gpuErrchk(cudaMemcpy(d_attribute_val_arr, h_attribute_val_arr,
                    attribute_val_arr_len * sizeof(int), cudaMemcpyHostToDevice));

        // for sorting information gain array
        gpuErrchk(cudaMemcpy(d_attribute_idx_arr, h_attribute_idx_arr, attribute_idx_arr_len *
                    sizeof(int), cudaMemcpyHostToDevice));


        dim3 grid(GROWING_TREE_COUNT, LEAF_COUNT_PER_TREE);
        thread_count = ATTRIBUTE_COUNT_PER_TREE * 2;

        compute_information_gain<<<grid, thread_count>>>(
                d_leaf_counters,
                d_is_leaf_active,
                d_tree_active_status,
                d_leaf_class,
                d_info_gain_vals,
                d_attribute_val_arr,
                ATTRIBUTE_COUNT_PER_TREE,
                ATTRIBUTE_COUNT_TOTAL,
                LEAF_COUNT_PER_TREE,
                CLASS_COUNT,
                LEAF_COUNTER_SIZE);

        cudaDeviceSynchronize();
        log_file << "compute_information_gain completed" << endl;



        gpuErrchk(cudaMemcpy(d_attribute_idx_arr, h_attribute_idx_arr, attribute_idx_arr_len *
                    sizeof(int), cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemset(d_node_split_decisions, 0, node_split_decisions_len * sizeof(int)));

        log_file << "\nlaunching compute_node_split_decisions kernel..." << endl;

        compute_node_split_decisions<<<GROWING_TREE_COUNT, LEAF_COUNT_PER_TREE>>>(
                d_info_gain_vals,
                d_is_leaf_active,
                d_tree_active_status,
                d_attribute_val_arr,
                d_attribute_idx_arr,
                d_node_split_decisions,
                ATTRIBUTE_COUNT_PER_TREE,
                r,
                delta,
                LEAF_COUNT_PER_TREE,
                d_samples_seen_count);

#if DEBUG

        // log info_gain_vals
        gpuErrchk(cudaMemcpy(h_info_gain_vals, d_info_gain_vals, info_gain_vals_len *
                    sizeof(float), cudaMemcpyDeviceToHost));

        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            log_file << "tree " << tree_idx << endl;
            int cur_tree_info_gain_vals_start_pos = tree_idx * LEAF_COUNT_PER_TREE *
                ATTRIBUTE_COUNT_PER_TREE * 2;

            for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
                int cur_info_gain_vals_start_pos = cur_tree_info_gain_vals_start_pos + leaf_idx *
                    ATTRIBUTE_COUNT_PER_TREE * 2;
                float *cur_info_gain_vals = h_info_gain_vals + cur_info_gain_vals_start_pos;

                for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
                    log_file << cur_info_gain_vals[i] << " ";
                }
                log_file << endl;
            }
            log_file << endl;
        }

#endif

        cudaDeviceSynchronize();
        log_file << "compute_node_split_decisions completed" << endl;


        log_file << "\nlaunching node_split kernel..." << endl;

        node_split<<<1, GROWING_TREE_COUNT>>>(
                d_decision_trees,
                d_tree_active_status,
                d_node_split_decisions,
                d_leaf_counters,
                d_leaf_class,
                d_leaf_back,
                d_attribute_val_arr,
                d_samples_seen_count,
                d_cur_node_count_per_tree,
                d_cur_leaf_count_per_tree,
                LEAF_COUNTER_SIZE,
                NODE_COUNT_PER_TREE,
                LEAF_COUNT_PER_TREE,
                ATTRIBUTE_COUNT_PER_TREE,
                ATTRIBUTE_COUNT_TOTAL,
                CLASS_COUNT);

        cudaDeviceSynchronize();

        log_file << "node_split completed" << endl;

        // for drift detection
        gpuErrchk(cudaMemcpy((void *) h_tree_error_count, (void *) d_tree_error_count,
                    tree_error_count_len * sizeof(int), cudaMemcpyDeviceToHost));

        int warning_tree_count = 0;
        int h_warning_tree_idx_arr[FOREGROUND_TREE_COUNT];
        int h_warning_tree_bg_idx_arr[FOREGROUND_TREE_COUNT];

        int drift_tree_count = 0;
        int h_drift_tree_idx_arr[FOREGROUND_TREE_COUNT];

        vector<char> target_state(cur_state);
        vector<int> warning_tree_id_list;
        vector<int> drift_tree_id_list;

        // warning/drift detection only on foreground trees
        // if accuracy decreases, reset the tree
        for (int tree_idx = 0; tree_idx < FOREGROUND_TREE_COUNT; tree_idx++) {
            ADWIN *warning_detector = warning_detectors[tree_idx];
            double old_error = warning_detector->getEstimation();
            bool error_change = warning_detector->setInput(h_tree_error_count[tree_idx]);

            if (error_change && old_error > warning_detector->getEstimation()) {
                error_change = false;
            }

            int bg_tree_pos = tree_idx + FOREGROUND_TREE_COUNT;

            // warning detected
            if (error_change) {
                warning_detectors[tree_idx] = new ADWIN((double) 0.001);

                // grow background tree
                if (h_tree_active_status[bg_tree_pos] == 2) {
                    // start growing if never grown
                    h_tree_active_status[bg_tree_pos] = 3;
                }

                target_state[forest_idx_to_tree_id[tree_idx]] = '2';
                warning_tree_id_list.push_back(forest_idx_to_tree_id[tree_idx]);

                h_warning_tree_idx_arr[warning_tree_count] = tree_idx;
                h_warning_tree_bg_idx_arr[warning_tree_count] = tree_idx
                    + FOREGROUND_TREE_COUNT;

                warning_tree_count++;
            }

            ADWIN *drift_detector = drift_detectors[tree_idx];
            old_error = drift_detector->getEstimation();
            error_change = drift_detector->setInput(h_tree_error_count[tree_idx]);

            if (error_change && old_error > drift_detector->getEstimation()) {
                // if error is decreasing, do nothing
                error_change = false;
            }

            if (!error_change) {
                continue;
            }

            // drift detected
            warning_detectors[tree_idx] = new ADWIN((double) 0.001);
            drift_detectors[tree_idx] = new ADWIN((double) 0.00001);

            h_drift_tree_idx_arr[drift_tree_count] = tree_idx;
            drift_tree_count++;

            drift_tree_id_list.push_back(forest_idx_to_tree_id[tree_idx]);
        }


        if (warning_tree_count > 0) {

            cout << endl
                << "ಠ_ಠ Warning detected at iter_count = " << iter_count << endl;
            cout << "warning tree forest_idx: ";
            for (int i = 0; i < warning_tree_count; i++) {
                cout << h_warning_tree_idx_arr[i] << " ";
            }

            cout << endl;
            // reset background trees
            gpuErrchk(cudaMemcpy(d_warning_tree_idx_arr, h_warning_tree_bg_idx_arr,
                        warning_tree_count * sizeof(int), cudaMemcpyHostToDevice));

            reset_tree<<<1, warning_tree_count>>>(
                    d_warning_tree_idx_arr,
                    d_decision_trees,
                    d_leaf_counters,
                    d_leaf_class,
                    d_leaf_back,
                    d_samples_seen_count,
                    d_cur_node_count_per_tree,
                    d_cur_leaf_count_per_tree,
                    d_tree_confusion_matrix,
                    NODE_COUNT_PER_TREE,
                    LEAF_COUNT_PER_TREE,
                    LEAF_COUNTER_SIZE,
                    leaf_counter_row_len,
                    confusion_matrix_size,
                    CLASS_COUNT);

            cudaDeviceSynchronize();
        }

        if (warning_tree_count > 0 || drift_tree_count > 0) {

            gpuErrchk(cudaMemcpy(h_decision_trees, d_decision_trees, TREE_COUNT
                        * NODE_COUNT_PER_TREE * sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_leaf_class, d_leaf_class, TREE_COUNT
                        * LEAF_COUNT_PER_TREE * sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_leaf_back, d_leaf_back, TREE_COUNT
                        * LEAF_COUNT_PER_TREE * sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_leaf_counters, d_leaf_counters,
                        ALL_LEAF_COUNTERS_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_cur_node_count_per_tree, d_cur_node_count_per_tree,
                        GROWING_TREE_COUNT * sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_cur_leaf_count_per_tree, d_cur_leaf_count_per_tree,
                        GROWING_TREE_COUNT * sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_samples_seen_count, d_samples_seen_count,
                        GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE * sizeof(int), cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(h_tree_confusion_matrix, d_tree_confusion_matrix,
                        TREE_COUNT * confusion_matrix_size * sizeof(int), cudaMemcpyDeviceToHost));

        }


        if (warning_tree_count > 0) {
            vector<char> closest_state;
            if  (state_transition_graph->is_stable) {
                closest_state = target_state;

                for (int warning_tree_id : warning_tree_id_list) {
                    int next_tree_id =
                        state_transition_graph->get_next_tree_id(warning_tree_id);

                    closest_state[warning_tree_id] = '0';
                    closest_state[next_tree_id] = '1';
                }

            } else {
                closest_state = state_queue->get_closest_state(target_state);
            }

            string closest_state_str(closest_state.begin(), closest_state.end());
            cout << "get_closest_state: " << closest_state_str << endl;

            if (closest_state.size() != 0) {
                for (int i = 0; i < cur_tree_pool_size; i++) {

                    if (tree_id_to_forest_idx[i] != -1) {
                        // tree already in forest
                        continue;
                    }

                    if (cur_state[i] == '1' && closest_state[i] == '0') {
                        // do nothing

                    } else if (cur_state[i] == '0' && closest_state[i] == '1') {


                        int next_avail_forest_idx;
                        if (next_empty_forest_idx.size() == 0) {
                            cout << "no next empty forest_idx" << endl;

                            candidate_t lru_candidate = forest_candidate_vec[0];
                            next_avail_forest_idx = lru_candidate.forest_idx;

                            forest_candidate_vec.erase(forest_candidate_vec.begin(),
                                    forest_candidate_vec.begin() + 1);

                        } else {
                            next_avail_forest_idx = next_empty_forest_idx.front();
                            next_empty_forest_idx.pop();
                        }

                        cout << "add tree_id" << i << " to forest_idx " <<
                            next_avail_forest_idx << endl;

                        if (next_avail_forest_idx < GROWING_TREE_COUNT
                                || next_avail_forest_idx >= TREE_COUNT) {

                            cout << "next_avail_forest_idx out of bound: " <<
                                next_avail_forest_idx << endl;
                            return 1;
                        }

                        candidate_t candidate = candidate_t(i, next_avail_forest_idx);
                        forest_candidate_vec.push_back(candidate);

                        tree_memcpy(&cpu_forest[i], &h_forest[next_avail_forest_idx], false);
                        h_tree_active_status[next_avail_forest_idx] = 5;
                        forest_idx_to_tree_id[next_avail_forest_idx] = i;
                        tree_id_to_forest_idx[i] = next_avail_forest_idx;

                        // reset candiate stats, so kappa becomes 0
                        // h_tree_error_count[next_avail_forest_idx] = INSTANCE_COUNT_PER_TREE;

                        int* candidate_confusion_matrix = h_tree_confusion_matrix
                            + next_avail_forest_idx * confusion_matrix_size;

                        memset(candidate_confusion_matrix, 0, confusion_matrix_size * sizeof(int));

                    }
                }
            }
        }


        if (drift_tree_count > 0) {

            cout << endl
                << "(╯°□°）╯︵ ┻━┻ drift detected at iter_count = " << iter_count << endl;

            cout << "drift tree forest_idx: ";
            for (int i = 0; i < drift_tree_count; i++) {
                cout << h_drift_tree_idx_arr[i] << " ";
            }
            cout << endl;

#if DEBUG
            cout << "tree active status: ";
            for (int i = 0; i < TREE_COUNT; i++) {
                cout << h_tree_active_status[i] << " ";
            }
            cout << endl;

            cout << "target_state: ";
            for (int i = 0; i < CPU_TREE_POOL_SIZE; i++) {
                cout << target_state[i] << " ";
            }
            cout << endl;


            for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
                int* cur_tree = h_decision_trees + tree_idx * NODE_COUNT_PER_TREE;
                int* cur_leaf_class = h_leaf_class + tree_idx *
                    LEAF_COUNT_PER_TREE;
                int* cur_leaf_back = h_leaf_back + tree_idx *
                    LEAF_COUNT_PER_TREE;
                cout << "tree " << tree_idx << ":" << endl;
                for (int node_idx = 0; node_idx < NODE_COUNT_PER_TREE; node_idx++) {
                    cout << cur_tree[node_idx] << ",";
                }
                cout << endl;
                for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
                    cout << cur_leaf_class[leaf_idx] << ",";
                }
                cout << endl;
                for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
                    cout << cur_leaf_back[leaf_idx] << ",";
                }
                cout << endl;
            }

#endif

            // calculate all kappa
            vector<candidate_t> forest_candidate_vec_copy = forest_candidate_vec;

            for (int i = 0; i < forest_candidate_vec_copy.size(); i++) {
                candidate_t candidate = forest_candidate_vec_copy[i];

                double accuracy = (INSTANCE_COUNT_PER_TREE
                        - h_tree_error_count[candidate.forest_idx])
                        / (double) INSTANCE_COUNT_PER_TREE;

                int* cur_confusion_matrix = h_tree_confusion_matrix + candidate.forest_idx
                    * confusion_matrix_size;

                forest_candidate_vec_copy[i].kappa = get_kappa(
                        cur_confusion_matrix,
                        CLASS_COUNT,
                        accuracy,
                        INSTANCE_COUNT_PER_TREE);
            }

            sort(forest_candidate_vec_copy.begin(), forest_candidate_vec_copy.end());

            vector<char> next_state(cur_state);

            for (int i = 0; i < drift_tree_id_list.size(); i++) {
                if (cur_tree_pool_size >= CPU_TREE_POOL_SIZE) {
                    // TODO
                    cout << "reached CPU_TREE_POOL_SIZE limit!" << endl;
                    return 1;
                }

                int tree_id = drift_tree_id_list[i];
                int forest_tree_idx = h_drift_tree_idx_arr[i];
                int forest_bg_tree_idx = forest_tree_idx + FOREGROUND_TREE_COUNT;

                cout << "tree_id: " << tree_id << endl;
                cout << "forest_tree_idx: " << forest_tree_idx << endl;
                cout << "forest_bg_tree_idx: " << forest_bg_tree_idx << endl;

                if (tree_id < 0 || tree_id >= CPU_TREE_POOL_SIZE) {
                    cout << "wrong tree_id" << endl;
                    return 1;
                }

                if (forest_tree_idx < 0 || forest_tree_idx >= FOREGROUND_TREE_COUNT) {
                    cout << "wrong forest_tree_idx" << endl;
                    return 1;
                }

                if (forest_bg_tree_idx < FOREGROUND_TREE_COUNT || forest_bg_tree_idx >=
                        GROWING_TREE_COUNT) {
                    cout << "wrong forest_bg_tree_idx" << endl;
                    return 1;
                }

                double fg_tree_accuracy = (INSTANCE_COUNT_PER_TREE
                        - h_tree_error_count[forest_tree_idx])
                        / (double) INSTANCE_COUNT_PER_TREE;

                int* cur_confusion_matrix = h_tree_confusion_matrix + forest_tree_idx
                    * confusion_matrix_size;

                double drift_tree_kappa = get_kappa(
                        cur_confusion_matrix,
                        CLASS_COUNT,
                        fg_tree_accuracy,
                        INSTANCE_COUNT_PER_TREE);

                // cout << "--------------drift kappa: " << drift_tree_kappa << endl;
                // cout << "---------fg_tree_accuracy: " << fg_tree_accuracy << endl;

                int forest_swap_tree_idx = forest_tree_idx;

                if (forest_candidate_vec_copy.size() > 0) {
                    // cout << "-----------------candidate kappa: " << cd_tree_kappa << endl;
                    // cout << "---------candidate_tree_accuracy: " << cd_tree_accuracy << endl;

                    candidate_t best_candidate =
                        forest_candidate_vec_copy[forest_candidate_vec_copy.size() - 1];

                    if (best_candidate.kappa - drift_tree_kappa > kappa_threshold) {
                        forest_swap_tree_idx = best_candidate.tree_id;
                        cout << "------------picked candidate tree: "
                            << best_candidate.tree_id << endl;


#if DEBUG
                        cout << "best_candiate.tree_id: " << best_candidate.tree_id << endl;
                        int* cur_tree = cpu_forest[best_candidate.tree_id].tree;
                        for (int node_idx = 0; node_idx < NODE_COUNT_PER_TREE; node_idx++) {
                            cout << cur_tree[node_idx] << ",";
                        }
                        cout << endl;
#endif
                    }
                }

                if (forest_swap_tree_idx == forest_tree_idx
                        && h_tree_active_status[forest_bg_tree_idx] == 3) {

                    double bg_tree_accuracy = (INSTANCE_COUNT_PER_TREE
                                - h_tree_error_count[forest_bg_tree_idx])
                                / (double) INSTANCE_COUNT_PER_TREE;

                    int* cur_bg_tree_confusion_matrix = h_tree_confusion_matrix
                        + forest_bg_tree_idx * confusion_matrix_size;

                    double bg_tree_kappa = get_kappa(
                            cur_bg_tree_confusion_matrix,
                            CLASS_COUNT,
                            bg_tree_accuracy,
                            INSTANCE_COUNT_PER_TREE);

                    // cout << "-----------------bg kappa: " << bg_tree_kappa << endl;
                    // cout << "---------bg_tree_accuracy: " << bg_tree_accuracy << endl;

                    if (bg_tree_kappa - drift_tree_kappa > 0.01) {
                        forest_swap_tree_idx = -1;
                    }
                }
                h_tree_active_status[forest_bg_tree_idx] = 2;


                if (forest_swap_tree_idx == forest_tree_idx) {
                    continue;
                }

                // put drift tree back to cpu tree pool
                tree_memcpy(&h_forest[forest_tree_idx], &cpu_forest[tree_id], true);
                tree_id_to_forest_idx[tree_id] = -1;
                forest_idx_to_tree_id[forest_tree_idx] = -1;

                if (forest_swap_tree_idx == -1) {
                    cout << "pick background tree" << endl;

                    // replace drift tree with its background tree
                    tree_memcpy(&h_forest[forest_bg_tree_idx], &h_forest[forest_tree_idx], true);

                    // add background tree to cpu_tree_pool
                    int new_tree_id = cur_tree_pool_size;
                    tree_memcpy(&h_forest[forest_bg_tree_idx], &cpu_forest[new_tree_id], true);

                    forest_idx_to_tree_id[forest_tree_idx] = new_tree_id;
                    tree_id_to_forest_idx[new_tree_id] = forest_tree_idx;

                    next_state[new_tree_id] = '1';
                    next_state[tree_id] = '0';

                    cur_tree_pool_size++;

                } else {

                    // find the best candidate and replace it with drift tree

                    candidate_t best_candidate =
                        forest_candidate_vec_copy[forest_candidate_vec_copy.size() - 1];

                    if (best_candidate.tree_id < 0 || best_candidate.tree_id >=
                            CPU_TREE_POOL_SIZE) {
                        cout << "incorrect best_candidate.tree_id" << endl;
                        return 1;
                    }

                    if (best_candidate.forest_idx < GROWING_TREE_COUNT || best_candidate.forest_idx >=
                            TREE_COUNT) {
                        cout << "incorrect best_candidate.forest_idx" << endl;
                        return 1;
                    }


                    // replace drift tree with its candidate tree
                    tree_memcpy(&cpu_forest[best_candidate.tree_id],
                            &h_forest[forest_tree_idx], true);

                    cout << "replace drift tree with candidate " << best_candidate.tree_id
                        << endl;

                    tree_id_to_forest_idx[best_candidate.tree_id] = forest_tree_idx;
                    forest_idx_to_tree_id[forest_tree_idx] = best_candidate.tree_id;

                    forest_idx_to_tree_id[best_candidate.forest_idx] = -1;

                    next_empty_forest_idx.push(best_candidate.forest_idx);
                    h_tree_active_status[best_candidate.forest_idx] = 4;

                    next_state[best_candidate.tree_id] = '1';
                    next_state[tree_id] = '0';


                    cout << "----------------->forest_candidate_vec: ";
                    for (int i = 0; i < forest_candidate_vec.size(); i++) {
                        cout << forest_candidate_vec[i].tree_id << ",";
                    }
                    cout << endl;
                    for (int i = 0; i < forest_candidate_vec.size(); i++) {
                        if (forest_candidate_vec[i].tree_id == best_candidate.tree_id) {
                            next_empty_forest_idx.push(forest_candidate_vec[i].forest_idx);
                            forest_candidate_vec.erase(forest_candidate_vec.begin() + i);
                            break;
                        }
                    }

                    forest_candidate_vec_copy.erase(forest_candidate_vec_copy.begin()
                            + forest_candidate_vec_copy.size() - 1);
                }

            }

#if DEBUG

            for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
                int* cur_tree = h_decision_trees + tree_idx * NODE_COUNT_PER_TREE;
                int* cur_leaf_class = h_leaf_class + tree_idx *
                    LEAF_COUNT_PER_TREE;
                int* cur_leaf_back = h_leaf_class + tree_idx *
                    LEAF_COUNT_PER_TREE;
                cout << "tree " << tree_idx << ":" << endl;
                for (int node_idx = 0; node_idx < NODE_COUNT_PER_TREE; node_idx++) {
                    cout << cur_tree[node_idx] << ",";
                }
                cout << endl;
                for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
                    cout << cur_leaf_class[leaf_idx] << ",";
                }
                cout << endl;
                for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
                    cout << cur_leaf_back[leaf_idx] << ",";
                }
                cout << endl;
            }
            cout << endl;

            cout << "CPU copied data: " << endl;
            for (int tree_idx = 0; tree_idx < cur_tree_pool_size; tree_idx++) {
                cout << "cpu tree " << tree_idx << ":" << endl;
                int* cur_cpu_tree = cpu_decision_trees + tree_idx * NODE_COUNT_PER_TREE;
                for (int i = 0; i < NODE_COUNT_PER_TREE; i++) {
                    cout << cur_cpu_tree[i] << ",";
                }
                cout << endl;
                int* cur_cpu_leaf_class = cpu_leaf_class + tree_idx * LEAF_COUNT_PER_TREE;
                for (int i = 0; i < LEAF_COUNT_PER_TREE; i++) {
                    cout << cur_cpu_leaf_class[i] << ",";
                }
                cout << endl;
            }

            cout << "forest_idx_to_tree_id: " << endl;
            for (int i = 0; i < FOREGROUND_TREE_COUNT; i++) {
                    cout << forest_idx_to_tree_id[i] << " ";
            }
            cout << endl;

#endif

            cur_state = next_state;
            state_queue->enqueue(cur_state);
            state_queue->to_string();
        }


        if (warning_tree_count > 0 || drift_tree_count > 0) {

            gpuErrchk(cudaMemcpy(d_decision_trees, h_decision_trees, NODE_COUNT_PER_TREE
                        * TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_leaf_class, h_leaf_class, TREE_COUNT
                        * LEAF_COUNT_PER_TREE * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_leaf_back, h_leaf_back, TREE_COUNT
                        * LEAF_COUNT_PER_TREE * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_leaf_counters, h_leaf_counters,
                        ALL_LEAF_COUNTERS_SIZE * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_cur_node_count_per_tree, h_cur_node_count_per_tree,
                        GROWING_TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_cur_leaf_count_per_tree, h_cur_leaf_count_per_tree,
                        GROWING_TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_samples_seen_count, h_samples_seen_count,
                        GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE * sizeof(int), cudaMemcpyHostToDevice));

            gpuErrchk(cudaMemcpy(d_tree_active_status, h_tree_active_status,
                        TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));
        }

        if (drift_tree_count > 0) {

            // TODO reset background trees only
            for (int i = 0; i < drift_tree_count; i++) {
                int bg_tree_forest_idx = h_drift_tree_idx_arr[i] + FOREGROUND_TREE_COUNT;
                h_drift_tree_idx_arr[i] = bg_tree_forest_idx;
                h_tree_active_status[bg_tree_forest_idx] = 2;
            }

            gpuErrchk(cudaMemcpy(d_drift_tree_idx_arr, h_drift_tree_idx_arr,
                        drift_tree_count * sizeof(int), cudaMemcpyHostToDevice));

            reset_tree<<<1, drift_tree_count>>>(
                    d_drift_tree_idx_arr,
                    d_decision_trees,
                    d_leaf_counters,
                    d_leaf_class,
                    d_leaf_back,
                    d_samples_seen_count,
                    d_cur_node_count_per_tree,
                    d_cur_leaf_count_per_tree,
                    d_tree_confusion_matrix,
                    NODE_COUNT_PER_TREE,
                    LEAF_COUNT_PER_TREE,
                    LEAF_COUNTER_SIZE,
                    leaf_counter_row_len,
                    confusion_matrix_size,
                    CLASS_COUNT);

            cudaDeviceSynchronize();
        }

        iter_count++;
    }

    log_file << "cur_tree_pool_size: " << cur_tree_pool_size << endl;
    log_file << "pattern matched: " << matched_pattern << endl;
    log_file << "\ntraining completed" << endl;

#if DEBUG

    int *h_decision_trees_log = (int*) malloc(NODE_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    gpuErrchk(cudaMemcpy(h_decision_trees_log, d_decision_trees, TREE_COUNT
                * NODE_COUNT_PER_TREE * sizeof(int), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(h_leaf_class, d_leaf_class, TREE_COUNT * LEAF_COUNT_PER_TREE *
                sizeof(int), cudaMemcpyDeviceToHost));

    int root_count = 0;
    for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
       cout << "Tree #" << tree_idx << endl;
       int *cur_decision_trees_log = h_decision_trees_log + tree_idx
           * NODE_COUNT_PER_TREE;
       int *cur_leaf_class = h_leaf_class + tree_idx * LEAF_COUNT_PER_TREE;


       for (int i = 0; i < NODE_COUNT_PER_TREE; i++) {
           int val = cur_decision_trees_log[i];
           if (IS_BIT_SET(val, 31) && val != -1) {
               if (i == 0) root_count++;
               int index = (val & (~(1 << 31)));
               val = cur_leaf_class[index];
               cout << "leaf:" << val << " ";
           } else {
               cout << val + 1 << " ";
           }
       }
       cout << endl;
    }

#endif

    cudaFree(d_decision_trees);
    cudaFree(d_reached_leaf_ids);
    cudaFree(d_leaf_class);
    cudaFree(d_leaf_back);
    cudaFree(d_leaf_counters);
    cudaFree(d_data);
    cudaFree(d_info_gain_vals);
    cudaFree(d_node_split_decisions);
    cudaFree(d_samples_seen_count);
    cudaFree(d_cur_node_count_per_tree);
    cudaFree(d_cur_leaf_count_per_tree);
    cudaFree(d_attribute_val_arr);
    cudaFree(d_attribute_idx_arr);
    cudaFree(d_confusion_matrix);
    cudaFree(d_tree_confusion_matrix);

    output_file.close();

    return 0;
}
