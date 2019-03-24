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

using namespace std;

#define EPS 1e-5
#define IS_BIT_SET(val, pos) (val & (1 << pos))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <typename T>
bool allocate_memory_on_device(T **arr, string arr_name, int count) {
    size_t memory_size = count * sizeof(T);
    cout << "\nAllocating " << memory_size << " bytes for " << arr_name << " on device..." << endl;

    cudaError_t err = cudaMalloc((void **) arr, memory_size); // allocate global memory on the device
    if (err != cudaSuccess) {
        cout << "error allocating memory for " << arr_name << " on device: " << memory_size << " bytes" << endl;
        return false;
    } else {
        cudaMemset((void **) arr, 0, memory_size);
        cout << "device: memory for " << arr_name << " allocated successfully." << endl;
        return true;
    }
}

double get_kappa(int *confusion_matrix, int class_count,  double accuracy, int sample_count) {
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
    int i;
    for (i = 0; i < k; i++) {
        reservoir[i] = i;
    }

    for (i = k; i < n; i++) {
        int j = rand() % i;

        if (j < k) reservoir[j] = i;
    }
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

__global__ void setup_kernel(curandState *state) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
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
        int max_node_count_per_tree,
        int max_leaf_count_per_tree,
        int leaf_counter_size,
        int leaf_counter_row_len,
        int class_count) {

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
    cur_samples_seen_count[0] = 0;

    int *cur_leaf_counter = leaf_counters + tree_idx * max_leaf_count_per_tree * leaf_counter_size;

    for (int k = 0; k < class_count + 2; k++) {
        for (int ij = 0; ij < leaf_counter_row_len; ij++) {
            cur_leaf_counter[k * leaf_counter_row_len + ij] = k == 1 ? 1 : 0;
        }
    }
}

__global__ void tree_traversal(
        int *decision_trees,
        bool *is_tree_active,
        int *data,
        int *reached_leaf_ids,
        int *leaf_class,
        int *correct_counter,
        int *samples_seen_count,
        int *forest_vote,
        int *forest_vote_idx_arr,
        int *weights,
        int *tree_error_count,
        int *confusion_matrix,
        int majority_class,
        int node_count_per_tree,
        int leaf_count_per_tree,
        int attribute_count_total,
        int class_count,
        curandState *state) {
    // <<<TREE_COUNT, INSTANCE_COUNT_PER_TREE>>>

    int tree_idx = blockIdx.x;

    if (!is_tree_active[tree_idx]) {
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
    int *cur_reached_leaf_ids = reached_leaf_ids + tree_idx * instance_count_per_tree;
    int *cur_leaf_class = leaf_class + tree_idx * leaf_count_per_tree;
    int *cur_samples_seen_count = samples_seen_count + tree_idx * leaf_count_per_tree;
    int *cur_forest_vote = forest_vote + instance_idx * class_count;

    int pos = 0;
    while (!IS_BIT_SET(cur_decision_tree[pos], 31)) {
        int attribute_id = cur_decision_tree[pos];
        pos = cur_data_line[attribute_id] == 0 ? get_left(pos) : get_right(pos);
    }

    int leaf_offset = (cur_decision_tree[pos] & (~(1 << 31)));
    cur_reached_leaf_ids[instance_idx] = leaf_offset;

    atomicAdd(&cur_samples_seen_count[leaf_offset], 1);

    int predicted_class = cur_leaf_class[leaf_offset];
    int actual_class = cur_data_line[attribute_count_total];

    if (pos == 0) {
        // predicted_class = get_rand(0, class_count - 1, state + thread_pos);
        predicted_class = majority_class;
    }

    if (predicted_class != actual_class) {
        atomicAdd(&tree_error_count[tree_idx], 1);
    }

    atomicAdd(&cur_forest_vote[predicted_class], 1);

    // online bagging
    int *cur_weights = weights + tree_idx * instance_count_per_tree;

    // curand library poisson is super slow!
    // cur_weights[instance_idx] = curand_poisson(state + thread_pos, 1.0);

    // prepare weights to be used in counter_increase kernel
    cur_weights[instance_idx] = poisson(1.0, state + thread_pos);
    // printf("==================================cur weight: %i\n", cur_weights[instance_idx]);

    __syncthreads();

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

__global__ void counter_increase(int *leaf_counters,
        int *reached_leaf_ids,
        int *data,
        int *attribute_val_arr,
        int *weights,
        int class_count,
        int attribute_count_per_tree,
        int attribute_count_total,
        int leaf_count_per_tree,
        int leaf_counter_size) {
    // gridDim: dim3(TREE_COUNT, INSTANCE_COUNT_PER_TREE)
    // blockDim: ATTRIBUTE_COUNT_PER_TREE
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

    int *cur_attribute_val_arr = attribute_val_arr + tree_idx * attribute_count_per_tree;

    // the counter start position corresponds to the leaf_id i.e. leaf offset
    int counter_start_pos = reached_leaf_id * leaf_counter_size + tree_idx *
        leaf_count_per_tree * leaf_counter_size;
    int *cur_leaf_counter = leaf_counters + counter_start_pos;
    // printf("leaf counter start pos is:  %i\n", counter_start_pos);

    int ij = cur_data[cur_attribute_val_arr[threadIdx.x]] + threadIdx.x * 2; // binary value 0 or 1
    int k = cur_data[attribute_count_total]; // class

    // int mask = cur_leaf_counter[attribute_count_per_tree * 2 + ij];
    int n_ijk_idx = (k + 2) * attribute_count_per_tree * 2 + ij;

    // atomicAdd(&cur_leaf_counter[ij], mask); // row 0
    // atomicAdd(&cur_leaf_counter[n_ijk_idx], mask);
    atomicAdd(&cur_leaf_counter[ij], cur_weight); // row 0
    atomicAdd(&cur_leaf_counter[n_ijk_idx], cur_weight);
}

__global__ void compute_information_gain(int *leaf_counters,
        int *leaf_class,
        float *info_gain_vals,
        int attribute_count_per_tree,
        int class_count,
        int leaf_counter_size) {
    // each leaf_counter is mapped to one block in the 1D grid
    // one thread uses one whole column per leaf counter
    // each block needs as many threads as twice number of the (binary) attributes

    // output: a vector with the attributes information gain values for all leaves in each of the trees
    // gridDim: dim3(TREE_COUNT, LEAF_COUNT_PER_TREE)
    // blockDim: attributes_per_tree * 2 (equal to a info_gain_vals per leaf)

    int block_id = blockIdx.y + blockIdx.x * gridDim.y;

    int thread_pos = threadIdx.x + block_id * blockDim.x;
    if (thread_pos >= gridDim.x * gridDim.y * blockDim.x) {
        return;
    }

    int tree_id = blockIdx.x;
    int leaf_id = blockIdx.y;

    int leaf_count_per_tree = gridDim.y;
    int leaf_counter_row_len = blockDim.x;

    int cur_tree_counters_start_pos= tree_id * leaf_count_per_tree * leaf_counter_size;
    int cur_leaf_counter_start_pos = cur_tree_counters_start_pos + leaf_id * leaf_counter_size;
    int *cur_leaf_counter = leaf_counters + cur_leaf_counter_start_pos;

    int cur_tree_info_gain_start_pos = tree_id * leaf_count_per_tree * leaf_counter_row_len;
    int cur_leaf_info_gain_start_pos = cur_tree_info_gain_start_pos + leaf_id *
        leaf_counter_row_len;
    float *cur_info_gain_vals = info_gain_vals + cur_leaf_info_gain_start_pos;

    int mask = cur_leaf_counter[leaf_counter_row_len + threadIdx.x];
    int a_ij = cur_leaf_counter[threadIdx.x];
    cur_info_gain_vals[threadIdx.x] = FLT_MAX;

    if (mask == 1) {
        // sum up a column
        float sum = 0.0;

        for (int i = 0; i < class_count; i++) {
            int a_ijk = cur_leaf_counter[threadIdx.x + (2 + i) * leaf_counter_row_len];

            // float param = a_ijk / a_ij; // TODO float division by zero returns INF
            // asm("max.f32 %0, %1, %2;" : "=f"(param) : "f"(param), "f"((float) 0.0));
            // sum += param * log(param);

            float param = 0.0;
            if (a_ijk != 0) { // && a_ij != 0) {
                param = (float) a_ijk / (float) a_ij;
            }

            float log_param = 0.0;
            if (abs(param) > EPS) {
                log_param = log(param);
            }

            sum += param * log_param;
        }

        cur_info_gain_vals[threadIdx.x] = -sum;
    }

    __syncthreads();

    float i_00 = 0.0, i_01 = 0.0;
    int i_idx = 0;

    if (threadIdx.x % 2 == 0) {
        i_00 = cur_info_gain_vals[threadIdx.x];
        i_01 = cur_info_gain_vals[threadIdx.x + 1];
        i_idx = (threadIdx.x >> 1);
    }

    __syncthreads();

    if (threadIdx.x % 2 == 0) {
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

    int *cur_leaf_class = leaf_class + tree_id * leaf_count_per_tree;
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

__global__ void compute_node_split_decisions(float *info_gain_vals,
        int *attribute_idx_arr,
        int *node_split_decisions,
        int attribute_count,
        float r,
        float delta,
        int *samples_seen_count) {
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
    int leaf_idx = threadIdx.x;
    int leaf_count_per_tree = blockDim.x;

    int cur_tree_attr_idx_start_pos = tree_idx * leaf_count_per_tree * attribute_count;
    int cur_leaf_attr_idx_start_pos = cur_tree_attr_idx_start_pos + leaf_idx * attribute_count;
    int *cur_attribute_idx_arr = attribute_idx_arr + cur_leaf_attr_idx_start_pos;

    int cur_tree_info_gain_start_pos = tree_idx * leaf_count_per_tree * attribute_count * 2;
    int cur_leaf_info_gain_start_pos = cur_tree_info_gain_start_pos + leaf_idx *
        attribute_count * 2;
    float *cur_info_gain_vals = info_gain_vals + cur_leaf_info_gain_start_pos;

    thrust::sort_by_key(thrust::seq,
            cur_info_gain_vals,
            cur_info_gain_vals + attribute_count,
            cur_attribute_idx_arr);

    float first_best = cur_info_gain_vals[0];
    float second_best = cur_info_gain_vals[1];

    float hoeffding_bound = compute_hoeffding_bound(r, delta, samples_seen_count[thread_pos]);

    int decision = 0;
    if (fabs(first_best - second_best) > hoeffding_bound) {
        // split on the best attribute
        decision |= (1 << 31);
        decision |= cur_attribute_idx_arr[0];
    }

    node_split_decisions[thread_pos] = decision;
}

__global__ void node_split(int *decision_trees,
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
        int class_count) {
    // <<<1, TREE_COUNT>>>
    // only launch one thread for each tree
    // to append new leaves at the end of the decision_tree array sequentially

    if (threadIdx.x >= blockDim.x) {
        return;
    }

    int tree_idx = threadIdx.x;
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
        int decision = cur_node_split_decisions[leaf_idx];
        cur_node_split_decisions[leaf_idx] = 0;

        int *cur_leaf_counter = cur_tree_leaf_counters + leaf_idx * counter_size_per_leaf;

        if (cur_node_count == max_node_count_per_tree) {
            // tree is full
            return;
        }

        if (!IS_BIT_SET(decision, 31)) {
            continue;
        }

        samples_seen_count[leaf_idx + tree_idx * max_leaf_count_per_tree] = 0;

        int attribute_id = (decision & ~(1 << 31));
        int cur_leaf_pos_in_tree = cur_leaf_back[leaf_idx];
        int cur_leaf_val = cur_decision_tree[cur_leaf_pos_in_tree];

        int old_leaf_id = (cur_leaf_val & ~(1 << 31));
        int new_leaf_id = cur_leaf_count;

        int left_leaf_pos = get_left(cur_leaf_pos_in_tree);
        int right_leaf_pos = get_right(cur_leaf_pos_in_tree);

        // cur_decision_tree[cur_leaf_pos_in_tree] = attribute_id;
        cur_decision_tree[cur_leaf_pos_in_tree] = cur_attribute_val_arr[attribute_id];

        cur_decision_tree[left_leaf_pos] = cur_leaf_val;
        cur_decision_tree[right_leaf_pos] = (1 << 31) | new_leaf_id;

        cur_leaf_back[old_leaf_id] = left_leaf_pos;
        cur_leaf_back[new_leaf_id] = right_leaf_pos;

        int left_max_class_code = 0;
        int left_max_count = cur_leaf_counter[attribute_count_per_tree * 2 * 2
            + attribute_id * 2];

        int right_max_class_code = 0;
        int right_max_count = cur_leaf_counter[attribute_count_per_tree * 2 * 2
            + attribute_id * 2 + 1];

        for (int k = 1; k < class_count; k++) {
            // left
            int cur_left_class_count = cur_leaf_counter[attribute_count_per_tree * 2 * (k + 2) +
                attribute_id * 2];
            if (cur_left_class_count > left_max_count) {
                left_max_count = cur_left_class_count;
                left_max_class_code = k;
            }

            // right
            int cur_right_class_count = cur_leaf_counter[attribute_count_per_tree * 2 * (k + 2) +
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
            int *cur_leaf_counter_row = cur_leaf_counter + attribute_count_per_tree * 2 * k;
            int *new_leaf_counter_row = new_leaf_counter + attribute_count_per_tree * 2 * k;

            if (k == 1) {
                for (int ij = 0; ij < attribute_count_per_tree * 2; ij++) {
                    if (ij == attribute_id * 2 || ij == attribute_id * 2 + 1) {
                        cur_leaf_counter_row[ij] = 0;
                    }

                    new_leaf_counter_row[ij] = cur_leaf_counter_row[ij];
                }

            } else {
                for (int ij = 0; ij < attribute_count_per_tree * 2; ij++) {
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
    int INSTANCE_COUNT_PER_TREE = 1000;
    string data_path = "data/covtype";
    string data_file_name = "covtype_binary_attributes.csv";
    bool ENABLE_BACKGROUND_TREES = false;

    int opt;
    while ((opt = getopt(argc, argv, "t:i:p:n:br")) != -1) {
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
            case 'b':
                ENABLE_BACKGROUND_TREES = true;
            case 'r':
                // Use a different seed value for each run
                srand(time(NULL));
                break;
        }
    }

    if (ENABLE_BACKGROUND_TREES) {
        TREE_COUNT *= 2;
    }

    cout << "TREE_COUNT = " << TREE_COUNT << endl
        << "INSTANCE_COUNT_PER_TREE = " << INSTANCE_COUNT_PER_TREE << endl;


    string output_path = data_path + "/result_gpu.txt";
    ofstream output_file;
    output_file.open(output_path);

    cout << endl;
    if (output_file.fail()) {
        cout << "Error opening output file at " << output_path << endl;
    } else {
        cout << "Writing output to " << output_path << endl;
    }


    // read data file
    string csv_path = data_path + "/" + data_file_name;
    ifstream data_file(csv_path);

    cout << endl;
    if (data_file) {
        cout << "Reading data file from " << csv_path << " succeeded." << endl;
    } else {
        cout << "Error reading file from " << csv_path << endl;
        return 1;
    }

    // prepare attributes
    string line;
    getline(data_file, line);

    const int ATTRIBUTE_COUNT_TOTAL = split(line, ",").size() - 1;
    const int ATTRIBUTE_COUNT_PER_TREE = (int) sqrt(ATTRIBUTE_COUNT_TOTAL);

    cout << "ATTRIBUTE_COUNT_TOTAL = " << ATTRIBUTE_COUNT_TOTAL << endl;
    cout << "ATTRIBUTE_COUNT_PER_TREE = " << ATTRIBUTE_COUNT_PER_TREE << endl;

    const unsigned int NODE_COUNT_PER_TREE = (1 << (ATTRIBUTE_COUNT_PER_TREE + 1)) - 1;
    const unsigned int LEAF_COUNT_PER_TREE = (1 << ATTRIBUTE_COUNT_PER_TREE);

    cout << "NODE_COUNT_PER_TREE = " << NODE_COUNT_PER_TREE << endl;
    cout << "LEAF_COUNT_PER_TREE = " << LEAF_COUNT_PER_TREE << endl;


    // read class/label file
    string class_path = data_path + "/labels.txt";
    ifstream class_file(class_path);

    cout << endl;
    if (class_file) {
        cout << "Reading class file from " << class_path << " succeeded." << endl;
    } else {
        cout << "Error reading class file from " << class_path << endl;
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
    cout << "CLASS_COUNT = " << CLASS_COUNT << endl;

    // hoeffding bound parameters
    float n_min = 1000;
    float delta = 0.05; // pow((float) 10.0, -7);
    float r = log2(CLASS_COUNT); // range of merit = log2(num_of_classes)

    cout << endl
        << "hoeffding bound parameters: " << endl
        << "n_min = " << n_min << endl
        << "delta = " << delta << endl
        << "r     = " << r     << endl;


    // init decision tree
    cout << "\nAllocating memory on host..." << endl;
    // void *allocated = malloc(NODE_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    void *allocated = calloc(NODE_COUNT_PER_TREE * TREE_COUNT, sizeof(int)); // TODO
    if (allocated == NULL) {
        cout << "host error: memory allocation for decision trees failed" << endl;
        return 1;
    }
    int *h_decision_trees = (int*) allocated;

    int *d_decision_trees;
    if (!allocate_memory_on_device(&d_decision_trees, "decision_trees", NODE_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }


#if DEBUG
    allocated = malloc(LEAF_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for leaf_class failed" << endl;
        return 1;
    }
    int *h_leaf_class = (int*) allocated; // stores the class for a given leaf

    allocated = malloc(LEAF_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for leaf_back failed" << endl;
        return 1;
    }
    int *h_leaf_back = (int*) allocated; // reverse pointer to map a leaf id to an offset in the tree array
#endif

    cout << "Init: set root as leaf for each tree in the forest..." << endl;
    for (int i = 0; i < TREE_COUNT; i++) {
        int *cur_decision_tree = h_decision_trees + i * NODE_COUNT_PER_TREE;
        cur_decision_tree[0] = (1 << 31);

        for (int j = 1; j < NODE_COUNT_PER_TREE; j++) {
            cur_decision_tree[j] = -1;
        }

    }

    gpuErrchk(cudaMemcpy(d_decision_trees, h_decision_trees, NODE_COUNT_PER_TREE * TREE_COUNT
                * sizeof(int), cudaMemcpyHostToDevice));

    // the offsets of leaves reached from tree traversal
    int *d_reached_leaf_ids;
    if (!allocate_memory_on_device(&d_reached_leaf_ids, "leaf_ids", INSTANCE_COUNT_PER_TREE * TREE_COUNT)) {
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

    // TODO: for testing only
    int leaf_counter_size = ATTRIBUTE_COUNT_PER_TREE * 2 * (CLASS_COUNT + 2);
    int all_leaf_counters_size = TREE_COUNT * LEAF_COUNT_PER_TREE * leaf_counter_size;

    // int *h_leaf_counters = (int*) malloc(all_leaf_counters_size * sizeof(int));
    int *h_leaf_counters = (int*) calloc(all_leaf_counters_size, sizeof(int));

    // init mask row
    for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
        int *cur_tree_leaf_counters = h_leaf_counters + tree_idx * LEAF_COUNT_PER_TREE *
            leaf_counter_size;
        for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
            int *cur_leaf_counter = cur_tree_leaf_counters + leaf_idx * leaf_counter_size;
            int *cur_leaf_counter_mask_row = cur_leaf_counter + ATTRIBUTE_COUNT_PER_TREE * 2;

            for (int k = 0; k < ATTRIBUTE_COUNT_PER_TREE * 2; k++) {
                cur_leaf_counter_mask_row[k] = 1;
            }
        }
    }

    int *d_leaf_counters;
    if (!allocate_memory_on_device(&d_leaf_counters, "leaf_counters", all_leaf_counters_size)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_leaf_counters, h_leaf_counters, all_leaf_counters_size * sizeof(int),
                cudaMemcpyHostToDevice));

    // TODO: h_info_gain_vals for testing only
    int info_gain_vals_len = TREE_COUNT * LEAF_COUNT_PER_TREE * ATTRIBUTE_COUNT_PER_TREE * 2;
    float *h_info_gain_vals = (float*) malloc(info_gain_vals_len * sizeof(float));

    float *d_info_gain_vals;
    if (!allocate_memory_on_device(&d_info_gain_vals, "info_gain_vals", info_gain_vals_len)) {
        return 1;
    }


    // actual selected attributes for each tree for counter_increase kernel
    int *h_attribute_val_arr;
    int *d_attribute_val_arr;
    int attribute_val_arr_len = TREE_COUNT * ATTRIBUTE_COUNT_PER_TREE;

    allocated = malloc(attribute_val_arr_len * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for h_attribute_val_arr failed" << endl;
    }
    h_attribute_val_arr = (int*) allocated;

    if (!allocate_memory_on_device(&d_attribute_val_arr, "attribute_val_arr",
                attribute_val_arr_len)) {
        return 1;
    }

    // select k random attributes for each tree
    // output_file << "\nAttributes selected per tree: " << endl;
    for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
        // output_file << "tree " << tree_idx << endl;

        int *cur_attribute_val_arr = h_attribute_val_arr + tree_idx * ATTRIBUTE_COUNT_PER_TREE;
        select_k_attributes(cur_attribute_val_arr, ATTRIBUTE_COUNT_TOTAL, ATTRIBUTE_COUNT_PER_TREE);

        // for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
        //     output_file << cur_attribute_val_arr[i] << " ";
        // }
        // output_file << endl;
    }

    gpuErrchk(cudaMemcpy(d_attribute_val_arr, h_attribute_val_arr,
                attribute_val_arr_len * sizeof(int), cudaMemcpyHostToDevice));


    // allocate memory for attribute indices on host for computing information gain
    int *h_attribute_idx_arr;
    int *d_attribute_idx_arr;
    int attribute_idx_arr_len = TREE_COUNT * LEAF_COUNT_PER_TREE * ATTRIBUTE_COUNT_PER_TREE;

    allocated = malloc(attribute_idx_arr_len * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for h_attribute_idx_arr failed" << endl;
        return 1;
    }
    h_attribute_idx_arr = (int*) allocated;

    if (!allocate_memory_on_device(&d_attribute_idx_arr, "attribute_idx_arr",
                attribute_idx_arr_len)) {
        return 1;
    }

    for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
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


    // allocate memory for node_split_decisions
    // unsigned int *h_node_split_decisions;
    int *d_node_split_decisions;
    int node_split_decisions_len = LEAF_COUNT_PER_TREE * TREE_COUNT;

    // allocated = malloc(node_split_decisions_len * sizeof(unsigned int));
    // if (allocated == NULL) {
    //     cout << "host error: memory allocation for h_node_split_decisions failed" << endl;
    //     return 1;
    // }
    // h_node_split_decisions = (unsigned int*) allocated;

    if (!allocate_memory_on_device(&d_node_split_decisions, "node_split_decisions",
                node_split_decisions_len)) {
        return 1;
    }

    int samples_seen_count_len = TREE_COUNT * LEAF_COUNT_PER_TREE;
    int *h_samples_seen_count = (int*) calloc(samples_seen_count_len, sizeof(int));
    int *d_samples_seen_count;
    if (!allocate_memory_on_device(&d_samples_seen_count, "samples_seen_count",
                samples_seen_count_len)) {
        return 1;
    }

    int h_cur_node_count_per_tree[TREE_COUNT];
    int *d_cur_node_count_per_tree;

    fill_n(h_cur_node_count_per_tree, TREE_COUNT, 1);

    if (!allocate_memory_on_device(&d_cur_node_count_per_tree, "cur_node_count_per_tree",
                TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_cur_node_count_per_tree, h_cur_node_count_per_tree,
                TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));

    int h_cur_leaf_count_per_tree[TREE_COUNT];
    int *d_cur_leaf_count_per_tree;

    fill_n(h_cur_leaf_count_per_tree, TREE_COUNT, 1);

    if (!allocate_memory_on_device(&d_cur_leaf_count_per_tree, "leaf_count_per_tree", TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_cur_leaf_count_per_tree, h_cur_leaf_count_per_tree,
                 TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));

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
    if (!allocate_memory_on_device(&d_weights, "weights", TREE_COUNT * INSTANCE_COUNT_PER_TREE)) {
        return 1;
    }

    // one drift detector per tree to monitor accuracy
    ADWIN *drift_detectors = new ADWIN[TREE_COUNT];

    int* h_tree_error_count = (int*) calloc(TREE_COUNT, sizeof(int));
    int* d_tree_error_count;
    if (!allocate_memory_on_device(&d_tree_error_count, "tree_error_count", TREE_COUNT)) {
        return 1;
    }

    int* d_reseted_tree_idx_arr;
    if (!allocate_memory_on_device(&d_reseted_tree_idx_arr, "reseted_tree_idx_arr", TREE_COUNT)) {
        return 1;
    }

    // pointer to the start of the background decision trees
    int *d_backgound_decision_trees = d_decision_trees + (TREE_COUNT >> 1) * NODE_COUNT_PER_TREE;

    bool *d_is_tree_active;
    if (!allocate_memory_on_device(&d_is_tree_active, "d_is_tree_active", TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemset(d_is_tree_active, true, (TREE_COUNT >> 1) * sizeof(bool)));

    bool *d_is_tree_active_bg = d_is_tree_active + (TREE_COUNT >> 1);
    gpuErrchk(cudaMemset(d_is_tree_active_bg, false, (TREE_COUNT >> 1) * sizeof(bool)));

    int confusion_matrix_size = CLASS_COUNT * CLASS_COUNT;
    int *h_confusion_matrix = (int*) malloc(confusion_matrix_size * sizeof(int));
    int *d_confusion_matrix;
    if (!allocate_memory_on_device(&d_confusion_matrix, "d_confusion_matrix",
                confusion_matrix_size)) {
        return 1;
    }


    cout << "\nInitializing training data arrays..." << endl;

    int data_len = INSTANCE_COUNT_PER_TREE * (ATTRIBUTE_COUNT_TOTAL + 1);
    int *h_data = (int*) malloc(data_len * sizeof(int));

    int *d_data;
    if (!allocate_memory_on_device(&d_data, "data", data_len)) {
        return 1;
    }

    vector<string> raw_data_row;

    int block_count;
    int thread_count;

    cout << endl << "=====Training Start=====" << endl;

    int h_correct_counter = 0;
    int *d_correct_counter;
    gpuErrchk(cudaMalloc((void **) &d_correct_counter, sizeof(int)));

    curandState *d_state;
    cudaMalloc(&d_state, TREE_COUNT * INSTANCE_COUNT_PER_TREE * sizeof(curandState));

    setup_kernel<<<TREE_COUNT, INSTANCE_COUNT_PER_TREE>>>(d_state);
    cudaDeviceSynchronize();

    int leaf_counter_row_len = ATTRIBUTE_COUNT_PER_TREE * 2;
    int iter_count = 1;
    double mean_accuracy = 0;

    output_file << "#iteration,accuracy,kappa" << endl;

    bool eof = false;

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
            cout << "\ntraining completed" << endl;
            break; // TODO
        }

        cout << endl << "=================iteration " << iter_count
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

        cout << "\nlaunching tree_traversal kernel..." << endl;

        block_count = TREE_COUNT;
        thread_count = INSTANCE_COUNT_PER_TREE;

        gpuErrchk(cudaMemset(d_correct_counter, 0, sizeof(int)));
        gpuErrchk(cudaMemset(d_tree_error_count, 0, TREE_COUNT * sizeof(int)));
        gpuErrchk(cudaMemset(d_confusion_matrix, 0, confusion_matrix_size * sizeof(int)));

        gpuErrchk(cudaMemset(d_forest_vote, 0, forest_vote_len * sizeof(int)));
        gpuErrchk(cudaMemcpy(d_forest_vote_idx_arr, h_forest_vote_idx_arr, forest_vote_len *
                    sizeof(int), cudaMemcpyHostToDevice));

        cout << "launching " << block_count * thread_count << " threads for tree_traversal" << endl;

        tree_traversal<<<block_count, thread_count>>>(
                d_decision_trees,
                d_is_tree_active,
                d_data,
                d_reached_leaf_ids,
                d_leaf_class,
                d_correct_counter,
                d_samples_seen_count,
                d_forest_vote,
                d_forest_vote_idx_arr,
                d_weights,
                d_tree_error_count,
                d_confusion_matrix,
                majority_class,
                NODE_COUNT_PER_TREE,
                LEAF_COUNT_PER_TREE,
                ATTRIBUTE_COUNT_TOTAL,
                CLASS_COUNT,
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
        cout << "tree_traversal completed" << endl;

        gpuErrchk(cudaMemcpy(&h_correct_counter, d_correct_counter, sizeof(int),
                    cudaMemcpyDeviceToHost));

        cout << "h_correct_counter: " << h_correct_counter << endl;
        double accuracy = (double) h_correct_counter / INSTANCE_COUNT_PER_TREE;
        mean_accuracy = (iter_count * mean_accuracy + accuracy) / (iter_count + 1);

        gpuErrchk(cudaMemcpy(h_confusion_matrix, d_confusion_matrix,
                    confusion_matrix_size * sizeof(int), cudaMemcpyDeviceToHost));

        double kappa = get_kappa(h_confusion_matrix, CLASS_COUNT, accuracy,
                INSTANCE_COUNT_PER_TREE);

        cout << "\n=================statistics" << endl
            << "accuracy: " << accuracy << endl
            << "mean accuracy: " << mean_accuracy << endl
            << "kappa: " << kappa << endl;

        output_file << iter_count * INSTANCE_COUNT_PER_TREE
            << "," << accuracy * 100
            << "," << kappa << endl;
            // << " " << mean_accuracy << endl;


        // for drift detection
        gpuErrchk(cudaMemcpy((void *) h_tree_error_count, (void *) d_tree_error_count,
                    TREE_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

        int reseted_tree_count = 0;
        int h_reseted_tree_idx_arr[TREE_COUNT];

        // if accuracy decreases, reset the tree
        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            ADWIN *estimation_error_weight = &drift_detectors[tree_idx];
            double old_error = estimation_error_weight->getEstimation();

            bool error_change = estimation_error_weight->setInput(h_tree_error_count[tree_idx]);
            h_tree_error_count[tree_idx] = 0; // reset host to copy back to device

            if (error_change && old_error > estimation_error_weight->getEstimation()) {
                // if error is decreasing, do nothing
                error_change = false;
            }

            if (!error_change) {
                continue;
            }

            h_reseted_tree_idx_arr[reseted_tree_count] = tree_idx;
            reseted_tree_count++;
        }

        if (reseted_tree_count > 0) {
            // output_file << "# change detected: " << reseted_tree_count << endl;

            gpuErrchk(cudaMemcpy(d_reseted_tree_idx_arr, h_reseted_tree_idx_arr, reseted_tree_count
                        * sizeof(int), cudaMemcpyHostToDevice));

            reset_tree<<<1, reseted_tree_count>>>(
                    d_reseted_tree_idx_arr,
                    d_decision_trees,
                    d_leaf_counters,
                    d_leaf_class,
                    d_leaf_back,
                    d_samples_seen_count,
                    d_cur_node_count_per_tree,
                    d_cur_leaf_count_per_tree,
                    NODE_COUNT_PER_TREE,
                    LEAF_COUNT_PER_TREE,
                    leaf_counter_size,
                    leaf_counter_row_len,
                    CLASS_COUNT);

            cudaDeviceSynchronize();
        }

#if DEBUG
        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            cout << "tree " << tree_idx << endl;
            int *cur_decision_tree = h_decision_trees + tree_idx * NODE_COUNT_PER_TREE;
            int *cur_leaf_class = h_leaf_class + tree_idx * LEAF_COUNT_PER_TREE;
            int *cur_samples_seen_count = h_samples_seen_count + tree_idx * LEAF_COUNT_PER_TREE;

            for (int i = 0; i < NODE_COUNT_PER_TREE; i++) {
                cout << cur_decision_tree[i] << " ";
            }
            cout << endl;

            for (int i = 0; i < LEAF_COUNT_PER_TREE; i++) {
                cout << cur_leaf_class[i] << " ";
            }
            cout << endl;

            cout << "samples seen count: " << endl;
            for (int i = 0; i < LEAF_COUNT_PER_TREE; i++) {
                cout << cur_samples_seen_count[i] << " ";
            }
            cout << endl;
        }
#endif


        cout << "\nlaunching counter_increase kernel..." << endl;

        counter_increase
            <<<dim3(TREE_COUNT, INSTANCE_COUNT_PER_TREE), ATTRIBUTE_COUNT_PER_TREE>>>(
                    d_leaf_counters,
                    d_reached_leaf_ids,
                    d_data,
                    d_attribute_val_arr,
                    d_weights,
                    CLASS_COUNT,
                    ATTRIBUTE_COUNT_PER_TREE,
                    ATTRIBUTE_COUNT_TOTAL,
                    LEAF_COUNT_PER_TREE,
                    leaf_counter_size);

        cudaDeviceSynchronize();
        cout << "counter_increase completed" << endl;

#if DEBUG
        gpuErrchk(cudaMemcpy(h_leaf_counters, d_leaf_counters, all_leaf_counters_size
                    * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_cur_leaf_count_per_tree, d_cur_leaf_count_per_tree, TREE_COUNT
                    * sizeof(int), cudaMemcpyDeviceToHost));


        cout << "counter_increase result: " << endl;
        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            cout << "tree " << tree_idx << endl;

            cout << "h_cur_leaf_count_per_tree is: " << h_cur_leaf_count_per_tree[tree_idx] << endl;
            int *cur_tree_leaf_counter = h_leaf_counters + tree_idx * LEAF_COUNT_PER_TREE
                * leaf_counter_size;

            for (int leaf_idx = 0; leaf_idx < h_cur_leaf_count_per_tree[tree_idx]; leaf_idx++) {
                int *cur_leaf_counter = cur_tree_leaf_counter + leaf_idx * leaf_counter_size;
                for (int k = 0; k < CLASS_COUNT + 2; k++) {
                    cout << "row " << k << ": ";
                    for (int ij = 0; ij < counter_row_len; ij++) {
                        cout << right << setw(8)
                            << cur_leaf_counter[k * counter_row_len + ij] << " ";
                    }
                    cout << endl;
                }
            }
            cout << endl;
        }
#endif

        cout << "\nlanuching compute_information_gain kernel..." << endl;

        // for sorting information gain array
        gpuErrchk(cudaMemcpy(d_attribute_idx_arr, h_attribute_idx_arr, attribute_idx_arr_len *
                    sizeof(int), cudaMemcpyHostToDevice));


        dim3 grid(TREE_COUNT, LEAF_COUNT_PER_TREE);
        thread_count = ATTRIBUTE_COUNT_PER_TREE * 2;

        compute_information_gain<<<grid, thread_count>>>(d_leaf_counters,
                d_leaf_class,
                d_info_gain_vals,
                ATTRIBUTE_COUNT_PER_TREE,
                CLASS_COUNT,
                leaf_counter_size);

        cudaDeviceSynchronize();
        cout << "compute_information_gain completed" << endl;


        cout << "\nlaunching compute_node_split_decisions kernel..." << endl;

        compute_node_split_decisions<<<TREE_COUNT, LEAF_COUNT_PER_TREE>>>(
                d_info_gain_vals,
                d_attribute_idx_arr,
                d_node_split_decisions,
                ATTRIBUTE_COUNT_PER_TREE,
                r,
                delta,
                d_samples_seen_count);

#if DEBUG
        // log info_gain_vals
        gpuErrchk(cudaMemcpy(h_info_gain_vals, d_info_gain_vals, info_gain_vals_len *
                    sizeof(float), cudaMemcpyDeviceToHost));

        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            cout << "tree " << tree_idx << endl;
            int cur_tree_info_gain_vals_start_pos = tree_idx * LEAF_COUNT_PER_TREE *
                ATTRIBUTE_COUNT_PER_TREE * 2;

            for (int leaf_idx = 0; leaf_idx < LEAF_COUNT_PER_TREE; leaf_idx++) {
                int cur_info_gain_vals_start_pos = cur_tree_info_gain_vals_start_pos + leaf_idx *
                    ATTRIBUTE_COUNT_PER_TREE * 2;
                float *cur_info_gain_vals = h_info_gain_vals + cur_info_gain_vals_start_pos;

                for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
                    cout << cur_info_gain_vals[i] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
#endif

        cudaDeviceSynchronize();
        cout << "compute_node_split_decisions completed" << endl;


        cout << "\nlaunching node_split kernel..." << endl;

        node_split<<<1, TREE_COUNT>>>(
                d_decision_trees,
                d_node_split_decisions,
                d_leaf_counters,
                d_leaf_class,
                d_leaf_back,
                d_attribute_val_arr,
                d_samples_seen_count,
                d_cur_node_count_per_tree,
                d_cur_leaf_count_per_tree,
                leaf_counter_size,
                NODE_COUNT_PER_TREE,
                LEAF_COUNT_PER_TREE,
                ATTRIBUTE_COUNT_PER_TREE,
                CLASS_COUNT);

        cout << "node_split completed" << endl;

        iter_count++;
    }

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

    output_file.close();

    return 0;
}
