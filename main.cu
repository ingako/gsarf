#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>
#include <iomanip>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

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
        cout << "device: memory for " << arr_name << " allocated successfully." << endl;
        return true;
    }
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

__device__ unsigned int get_left(unsigned int index) {
    return 2 * index + 1; 
}

__device__ unsigned int get_right(unsigned int index) {
    return 2 * index + 2;
}

__global__ void tree_traversal(int *decision_trees, 
        int *data,
        int *leaf_ids,
        int *leaf_class,
        int *correct_counter,
        int *samples_seen_count,
        int leaf_count_per_tree,
        int attribute_count) {
    // <<<TREE_COUNT, INSTANCE_COUNT_PER_TREE>>>
    // int thread_pos = threadIdx.x + blockIdx.x * blockDim.x;

    int tree_idx = blockIdx.x;
    int instance_count_per_tree = blockDim.x;

    int thread_pos = threadIdx.x + tree_idx * instance_count_per_tree;
    if (thread_pos >= blockDim.x * gridDim.x) {
        return;
    }

    int tree_data_start_idx = tree_idx * instance_count_per_tree * (attribute_count + 1); // tree_idx * instance_count_per_tree * data_row_len
    int instance_data_start_idx = threadIdx.x * (attribute_count + 1) + tree_data_start_idx;

    int *cur_data_line = data + instance_data_start_idx; 

    int pos = 0;
    while (!IS_BIT_SET(decision_trees[pos], 31)) {
        int attribute_id = decision_trees[pos];
        pos = cur_data_line[attribute_id] == 0 ? get_left(pos) : get_right(pos);
    }

    int leaf_offset = (decision_trees[pos] & (~(1 << 31)));

    // leaf_ids[TREE_COUNT * INSTANCE_COUNT_PER_TREE] 
    leaf_ids[thread_pos] = leaf_offset; 

    // printf("class code: %i  actual_class: %i   class_idx: %i\n", leaf_class_code,
    //        cur_data_line[attribute_count], data_start_idx + attribute_count);

    atomicAdd(&samples_seen_count[leaf_offset + tree_idx * leaf_count_per_tree], 1);

    // TODO test parallel reduction
    if (leaf_class[leaf_offset] == cur_data_line[attribute_count]) {
        atomicAdd(correct_counter, 1);
    }
}

__global__ void counter_increase(int *leaf_counters, 
        int *leaf_ids,
        int *data,
        int class_count,
        int attribute_count) {
    // gridDim: dim3(TREE_COUNT, INSTANCE_COUNT_PER_TREE)
    // blockDim: ATTRIBUTE_COUNT_PER_TREE
    // increment both n_ij (at row 0) and n_ijk (at row k)

    // input: an array of leaf_ids (offset) and leaf_classes built from tree_traversal

    // Each leaf counter is represented by a block and uses one thread for each attribute i and
    // value j (i.e. one thread per column) 
    //
    // Row 0 stores the total number of times value n_ij appeared.
    // Row 1 is a mask that keeps track of which attributes have been already used in internal nodes
    // along the path.
    // Row 2 and onwards stores partial counters n_ijk for each class k.

    // int block_id = blockIdx.y + blockIdx.x * gridDim.x; // instance_idx

    int block_id = blockIdx.y + blockIdx.x * gridDim.y;

    int thread_pos = threadIdx.x + block_id * blockDim.x;
    if (thread_pos >= gridDim.x * gridDim.y * blockDim.x) {
        return;
    }
    // printf("check mask row for leaf 0: %i\n", leaf_counters[6]);

    int leaf_id = leaf_ids[block_id];
    int *cur_data = data + block_id * (attribute_count + 1);

    // the counter start position corresponds to the leaf_id i.e. leaf offset
    int counter_start_pos = leaf_id * attribute_count * 2 * (class_count + 2); // TODO TREE_COUNT
    int *cur_leaf_counter = leaf_counters + counter_start_pos;
    // printf("leaf counter start pos is:  %i\n", counter_start_pos);

    int ij = cur_data[threadIdx.x] + threadIdx.x * 2; // binary value 0 or 1
    int k = cur_data[attribute_count];

    int mask = cur_leaf_counter[attribute_count * 2 + ij];
    int n_ijk_idx = (k + 2) * attribute_count * 2 + ij;

    atomicAdd(&cur_leaf_counter[ij], mask); // row 0
    atomicAdd(&cur_leaf_counter[n_ijk_idx], mask);
}

__global__ void compute_information_gain(int *leaf_counters, 
        float *info_gain_vals, 
        int class_count) {
    // each leaf_counter is mapped to one block in the 1D grid
    // each block needs as many threads as twice number of the (binary) attributes

    // output: a vector with the attributes information gain  values for all leaves in each of the trees
    // gridDim: dim3(TREE_COUNT, LEAF_COUNT_PER_TREE)
    // blockDim: attributes_per_tree * 2 (equal to a info_gain_vals per leaf)

    int thread_pos = threadIdx.x + gridDim.x * blockIdx.x;
    if (thread_pos >= gridDim.x * blockDim.x) {
        return;
    }

    int tree_id = blockIdx.x;
    int leaf_id = blockIdx.y;

    int leaf_count_per_tree = gridDim.y;
    int leaf_counter_row_len = blockDim.x;
    int leaf_counter_size = blockDim.x * (class_count + 2);

    int cur_tree_start_pos = tree_id * leaf_count_per_tree * leaf_counter_size;
    int cur_leaf_start_pos = cur_tree_start_pos + leaf_id * leaf_counter_size;
    int *cur_leaf_counter = leaf_counters + cur_leaf_start_pos;

    int cur_tree_info_gain_start_pos = tree_id * leaf_count_per_tree * leaf_counter_row_len;
    int cur_leaf_info_gain_start_pos = cur_tree_info_gain_start_pos + leaf_id *
        leaf_counter_row_len;
    float *cur_info_gain_vals = info_gain_vals + cur_leaf_info_gain_start_pos; // TODO


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
            // sum += -(param) * log(param);

            float param = 0.0;
            if (a_ijk != 0 && a_ij != 0) {
                param = (float) a_ijk / a_ij;
            }

            float log_param = 0.0;
            if (abs(param) > EPS) {
                log_param = log(param);
            }

            sum += -(param) * log_param;
        }

        cur_info_gain_vals[threadIdx.x] = -sum;
    }


    __syncthreads();

    float i_00 = 0, i_01 = 0;
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
}

// hoeffding bound
// providing an upper bound on the probability that the sum of a sample of independent random
// variables deviates from its expected value
// 
// range: range of the random variable
// confidence: desired probability of the estimate not being within the expected value
// n: the number of examples collected at the node
__device__ float compute_hoeffding_bound(float range, float confidence, float n) {
    return sqrt(((range * range) * log(1.0 / confidence)) / (2.0 * n));
}

__global__ void compute_node_split_decisions(float *info_gain_vals, 
        int *attribute_idx_arr,
        unsigned int *node_split_decisions, 
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

    unsigned int decision = 0;
    if (fabs(first_best - second_best) > hoeffding_bound) {
        // split on the best attribute
        decision |= (1 << 31);
        decision |= cur_attribute_idx_arr[0];
    }

    node_split_decisions[thread_pos] = decision;
}

__global__ void node_split(int *decision_trees,
        unsigned int *node_split_decisions,
        int *leaf_counters,
        int *leaf_class,
        int *leaf_back,
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

    unsigned int *cur_node_split_decisions = node_split_decisions + tree_idx *
        max_leaf_count_per_tree;

    int *cur_tree_leaf_counters = leaf_counters + 
        tree_idx * max_leaf_count_per_tree * counter_size_per_leaf;

    int *cur_leaf_back = leaf_back + tree_idx * max_leaf_count_per_tree;

    for (int leaf_idx = 0; leaf_idx < max_leaf_count_per_tree; leaf_idx++) {
        unsigned int decision = cur_node_split_decisions[leaf_idx];
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

        cur_decision_tree[cur_leaf_pos_in_tree] = attribute_id;

        cur_decision_tree[left_leaf_pos] = cur_leaf_val;
        cur_decision_tree[right_leaf_pos] = (1 << 31) | new_leaf_id;

        leaf_back[old_leaf_id] = left_leaf_pos;
        leaf_back[new_leaf_id] = right_leaf_pos;

        // TODO only work for binary class case
        int class0_count = cur_leaf_counter[attribute_count_per_tree * 2 * 2 + attribute_id * 2];
        int class1_count = cur_leaf_counter[attribute_count_per_tree * 2 * 3 + attribute_id * 2];

        int left_child_class = 0, right_child_class = 1;
        if (class0_count < class1_count) {
            left_child_class = 1;
            right_child_class = 0;
        }

        leaf_class[old_leaf_id] = left_child_class;
        leaf_class[new_leaf_id] = right_child_class;


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

int main(void) {
    const int TREE_COUNT = 1;
    cout << "Number of decision trees: " << TREE_COUNT << endl;

    const int INSTANCE_COUNT_PER_TREE = 1024;
    cout << "Instance count per tree: " << INSTANCE_COUNT_PER_TREE << endl;

    // hoeffding bound parameters
    float n_min = TREE_COUNT * INSTANCE_COUNT_PER_TREE;
    float delta = 0.05; // pow((float) 10.0, -7);
    float r = 1;
    cout << "hoeffding bound parameters: " << endl
        << "n_min: " << n_min << endl
        << "delta: " << delta << endl
        << "r    : " << r     << endl;

    // Use a different seed value for each run
    // srand(time(NULL));

    cout << "Reading class file..." << endl; 
    ifstream class_file("data/random-tree/labels.txt");
    string class_line;

    // init mapping between class and code
    map<string, int> class_code_map;
    map<int, string> code_class_map;

    vector<string> class_arr = split(class_line, " ");
    string code_str, class_str;

    int line_count = 0;
    while (class_file >> code_str >> class_str) {
        int code_int = atoi(code_str.c_str());
        class_code_map[class_str] = code_int;
        code_class_map[code_int] = class_str;
        line_count++;
    }
    const int CLASS_COUNT = line_count; 
    cout << "Number of classes: " << CLASS_COUNT << endl;

    // prepare attributes
    std::ifstream file("data/random-tree/synthetic_with_noise.csv");
    string line;

    getline(file, line);
    // const int ATTRIBUTE_COUNT_TOTAL = split_attributes(line, ',').size() - 2; // for activity-recognition dataset
    const int ATTRIBUTE_COUNT_TOTAL = split(line, ",").size() - 1;
    const int ATTRIBUTE_COUNT_PER_TREE = (int) sqrt(ATTRIBUTE_COUNT_TOTAL);

    cout << "Attribute count total: " << ATTRIBUTE_COUNT_TOTAL << endl;
    cout << "Attribute count per tree: " << ATTRIBUTE_COUNT_PER_TREE << endl;

    const unsigned int NODE_COUNT_PER_TREE = (1 << (ATTRIBUTE_COUNT_PER_TREE + 1));
    const unsigned int LEAF_COUNT_PER_TREE = (1 << ATTRIBUTE_COUNT_PER_TREE);

    cout << "NODE_COUNT_PER_TREE: " << NODE_COUNT_PER_TREE << endl;
    cout << "LEAF_COUNT_PER_TREE: " << LEAF_COUNT_PER_TREE << endl;

    size_t memory_size;

    // select k random attributes for each tree
    int attribute_arr[TREE_COUNT * ATTRIBUTE_COUNT_PER_TREE];
    for (int i = 0; i < TREE_COUNT; i++) {
        select_k_attributes(attribute_arr + i * ATTRIBUTE_COUNT_PER_TREE, 
                ATTRIBUTE_COUNT_TOTAL, ATTRIBUTE_COUNT_PER_TREE);
    }

    cout << "\nAttributes selected per tree: " << endl;
    for (int i = 0; i < TREE_COUNT; i++) {
        cout << "tree " << i << endl;
        for (int j = 0; j < ATTRIBUTE_COUNT_PER_TREE; j++) {
            cout << attribute_arr[i * ATTRIBUTE_COUNT_PER_TREE + j] << " ";
        }
        cout << endl;
    }

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

    allocated = malloc(LEAF_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for leaf_class failed" << endl;
        return 1;
    }
    int *h_leaf_class = (int*) allocated; // stores the class for a given leaf
    int *d_leaf_class;

    allocated = malloc(LEAF_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for leaf_back failed" << endl;
        return 1;
    }
    int *h_leaf_back = (int*) allocated; // reverse pointer to map a leaf id to an offset in the tree array
    int *d_leaf_back;

    cout << "Init: set root as leaf for each tree in the forest..." << endl;
    for (int i = 0; i < TREE_COUNT; i++) {
        h_decision_trees[i * NODE_COUNT_PER_TREE] = (1 << 31); // init root node
    }

    cout << "\nAllocating memory on device..." << endl;

    if (!allocate_memory_on_device(&d_decision_trees, "decision_trees", NODE_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_decision_trees, h_decision_trees, NODE_COUNT_PER_TREE * TREE_COUNT 
                * sizeof(int), cudaMemcpyHostToDevice));

    // the offsets of leaves reached from tree traversal
    int *d_leaf_ids;
    if (!allocate_memory_on_device(&d_leaf_ids, "leaf_ids", INSTANCE_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }

    if (!allocate_memory_on_device(&d_leaf_class, "leaf_class", LEAF_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }

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
    unsigned int *h_node_split_decisions;
    unsigned int *d_node_split_decisions;
    int node_split_decisions_len = LEAF_COUNT_PER_TREE * TREE_COUNT;

    allocated = malloc(node_split_decisions_len * sizeof(unsigned int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for h_node_split_decisions failed" << endl;
        return 1;
    }
    h_node_split_decisions = (unsigned int*) allocated;

    if (!allocate_memory_on_device(&d_node_split_decisions, "node_split_decisions", 
                node_split_decisions_len)) {
        return 1;
    }

    int samples_seen_count_len = TREE_COUNT * LEAF_COUNT_PER_TREE;
    int *h_samples_seen_count = (int*) malloc(samples_seen_count_len * sizeof(int));
    int *d_samples_seen_count;
    if (!allocate_memory_on_device(&d_samples_seen_count, "samples_seen_count",
                samples_seen_count_len)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy((void *) d_samples_seen_count, (void *) h_samples_seen_count, 
                samples_seen_count_len * sizeof(int), cudaMemcpyHostToDevice));

    int h_cur_node_count_per_tree[TREE_COUNT] = { 1 };
    int *d_cur_node_count_per_tree;
    if (!allocate_memory_on_device(&d_cur_node_count_per_tree, "cur_node_count_per_tree", TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_cur_node_count_per_tree, h_cur_node_count_per_tree, 
                TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));

    int h_cur_leaf_count_per_tree[TREE_COUNT] = { 1 };
    int *d_cur_leaf_count_per_tree;
    if (!allocate_memory_on_device(&d_cur_leaf_count_per_tree, "leaf_count_per_tree", TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_cur_leaf_count_per_tree, h_cur_leaf_count_per_tree, 
                TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));


    cout << "\nInitializing training data arrays..." << endl;

    int data_len = TREE_COUNT * INSTANCE_COUNT_PER_TREE * (ATTRIBUTE_COUNT_PER_TREE + 1);
    int *h_data = (int*) malloc(data_len * sizeof(int));

    int *d_data;
    if (!allocate_memory_on_device(&d_data, "data", data_len)) {
        return 1;
    }

    vector<string> raw_data_row;
    int data_start_idx;

    int block_count;
    int thread_count;

    cout << endl << "=====Training Start=====" << endl;

    int h_correct_counter = 0;
    int *d_correct_counter;
    cudaMalloc((void **) &d_correct_counter, sizeof(int));

    bool eof = false;

    while (!eof) {

        cout << "preparing data..." << endl;
        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            int *cur_attribute_arr = attribute_arr + tree_idx * ATTRIBUTE_COUNT_PER_TREE;
            int *cur_tree_data_rows = h_data + (ATTRIBUTE_COUNT_PER_TREE + 1) * 
                INSTANCE_COUNT_PER_TREE * tree_idx;

            for (int instance_idx = 0; instance_idx < INSTANCE_COUNT_PER_TREE; instance_idx++) {
                if (!getline(file, line)) {
                    eof = true;
                    break;
                }

                raw_data_row = split(line, ",");

                int *cur_data_row = cur_tree_data_rows + (ATTRIBUTE_COUNT_PER_TREE + 1) * instance_idx;

                for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
                    int attribute_code = cur_attribute_arr[i];
                    int val = strcmp(raw_data_row[attribute_code].c_str(), (const char*) "value1") == 0 ? 0 : 1;

                    cur_data_row[i] = val;

                    // cout << "a" << attribute_arr[attribute_offset] << ":";
                    // cout << h_data[data_start_idx + i] << " ";
                }
                // cout << endl;

                cur_data_row[ATTRIBUTE_COUNT_PER_TREE] = class_code_map[raw_data_row[raw_data_row.size() - 1]]; // class
                // cout << "h_data_class_idx: " << data_start_idx + ATTRIBUTE_COUNT_PER_TREE << endl;
                // cout << "h_data class: " << h_data[data_start_idx + ATTRIBUTE_COUNT_PER_TREE] << endl;
            }

            if (eof) {
                break; // TODO
            }
        }
        if (eof) break; // TODO

        gpuErrchk(cudaMemcpy((void *) d_data, (void *) h_data, data_len * sizeof(int), cudaMemcpyHostToDevice));

        cout << "\nlaunching tree_traversal kernel..." << endl;

        block_count = TREE_COUNT;
        thread_count = INSTANCE_COUNT_PER_TREE;

        cudaMemset(d_correct_counter, 0, sizeof(int));

        cout << "launching " << block_count * thread_count << " threads for tree_traversal" << endl;

        tree_traversal<<<block_count, thread_count>>>(d_decision_trees,
                d_data,
                d_leaf_ids,
                d_leaf_class,
                d_correct_counter,
                d_samples_seen_count,
                LEAF_COUNT_PER_TREE,
                ATTRIBUTE_COUNT_PER_TREE);

        gpuErrchk(cudaMemcpy(h_decision_trees, d_decision_trees, TREE_COUNT * NODE_COUNT_PER_TREE *
                    sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaMemcpy((void *) h_samples_seen_count, (void *) d_samples_seen_count, 
                    samples_seen_count_len * sizeof(int), cudaMemcpyDeviceToHost));

        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            cout << "tree " << tree_idx << endl;
            int *cur_decision_tree = h_decision_trees + tree_idx * NODE_COUNT_PER_TREE;
            int *cur_samples_seen_count = h_samples_seen_count + tree_idx * LEAF_COUNT_PER_TREE;

            for (int i = 0; i < NODE_COUNT_PER_TREE; i++) {
                cout << cur_decision_tree[i] << " ";
            }
            cout << endl;

            cout << "samples seen count: " << endl;
            for (int i = 0; i < LEAF_COUNT_PER_TREE; i++) {
                cout << cur_samples_seen_count[i] << " ";
            }
            cout << endl;
        }

        cout << "tree_traversal completed" << endl;

        gpuErrchk(cudaMemcpy(&h_correct_counter, d_correct_counter, sizeof(int), cudaMemcpyDeviceToHost));
        cout << "h_correct_counter: " << h_correct_counter << endl;
        double accuracy = (double) h_correct_counter / (INSTANCE_COUNT_PER_TREE * TREE_COUNT);
        cout << "=============== accuracy: " << accuracy << endl;

        cout << "\nlaunching counter_increase kernel..." << endl;

        counter_increase
            <<<dim3(TREE_COUNT, INSTANCE_COUNT_PER_TREE), ATTRIBUTE_COUNT_PER_TREE>>>(
                    d_leaf_counters,
                    d_leaf_ids,
                    d_data,
                    CLASS_COUNT,
                    ATTRIBUTE_COUNT_PER_TREE);

        cudaDeviceSynchronize();

        // TODO: for testing only
        gpuErrchk(cudaMemcpy(h_leaf_counters, d_leaf_counters, all_leaf_counters_size 
                    * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(h_cur_leaf_count_per_tree, d_cur_leaf_count_per_tree, TREE_COUNT 
                    * sizeof(int), cudaMemcpyDeviceToHost));

        cout << "counter_increase result: " << endl;

        int row_len = ATTRIBUTE_COUNT_PER_TREE * 2;
        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            cout << "tree " << tree_idx << endl;

            cout << "h_cur_leaf_count_per_tree is: " << h_cur_leaf_count_per_tree[tree_idx] << endl;
            int *cur_tree_leaf_counter = h_leaf_counters + tree_idx * LEAF_COUNT_PER_TREE
                * leaf_counter_size;

            for (int leaf_idx = 0; leaf_idx < h_cur_leaf_count_per_tree[tree_idx]; leaf_idx++) {
                int *cur_leaf_counter = cur_tree_leaf_counter + leaf_idx * leaf_counter_size;

                for (int k = 0; k < CLASS_COUNT + 2; k++) {
                    cout << "row " << k << ": ";
                    for (int ij = 0; ij < row_len; ij++) {
                        cout << right << setw(8) << cur_leaf_counter[k * row_len + ij] << " ";
                    }
                    cout << endl;
                }
            }
            cout << endl;
        }


        cout << "\nlanuching compute_information_gain kernel..." << endl;

        dim3 grid(TREE_COUNT, LEAF_COUNT_PER_TREE);
        thread_count = ATTRIBUTE_COUNT_PER_TREE * 2;

        compute_information_gain<<<grid, thread_count>>>(d_leaf_counters,
                d_info_gain_vals,
                CLASS_COUNT);

        cout << "compute_information_gain completed" << endl;

        cout << "\nlaunching compute_node_split_decisions kernel..." << endl;

        gpuErrchk(cudaMemcpy(d_attribute_idx_arr, h_attribute_idx_arr, attribute_idx_arr_len *
                    sizeof(int), cudaMemcpyHostToDevice));

        compute_node_split_decisions<<<TREE_COUNT, LEAF_COUNT_PER_TREE>>>(d_info_gain_vals,
                d_attribute_idx_arr,
                d_node_split_decisions,
                ATTRIBUTE_COUNT_PER_TREE,
                r,
                delta,
                d_samples_seen_count);
        
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

        cout << "compute_node_split_decisions completed" << endl;

        cout << "\nlaunching node_split kernel..." << endl;

        node_split<<<1, TREE_COUNT>>>(d_decision_trees,
                d_node_split_decisions,
                d_leaf_counters,
                d_leaf_class,
                d_leaf_back,
                d_samples_seen_count,
                d_cur_node_count_per_tree,
                d_cur_leaf_count_per_tree,
                leaf_counter_size,
                NODE_COUNT_PER_TREE,
                LEAF_COUNT_PER_TREE,
                ATTRIBUTE_COUNT_PER_TREE,
                CLASS_COUNT);

        cout << "node_split completed" << endl;

        // break; // TODO
    }

    cudaFree(d_decision_trees);
    cudaFree(d_leaf_ids);
    cudaFree(d_leaf_class);
    cudaFree(d_leaf_back);
    cudaFree(d_leaf_counters);
    cudaFree(d_data);
    cudaFree(d_info_gain_vals);
    cudaFree(d_node_split_decisions);

    return 0;
}
