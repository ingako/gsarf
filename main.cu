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
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

using namespace std;

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
    if (err) {
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

__device__ bool is_leaf(unsigned int node) {
    return ((node >> 31) & 1) == 1;
}

__device__ unsigned int get_left(unsigned int index) {
    return 2 * index + 1; 
}

__device__ unsigned int get_right(unsigned int index) {
    return 2 * index + 2;
}

__global__ void tree_traversal(int *decision_trees, 
        int *attribute_arr, 
        int *data,
        int *leaf_class,
        int *leaf_back,
        int *correct_counter,
        int attribute_count) {
    // <<<TREE_COUNT, INSTANCE_COUNT_PER_TREE>>>
    int thread_pos = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_pos >= blockDim.x * gridDim.x) {
        return;
    }

    int tree_data_start_idx = blockIdx.x * blockDim.x * (attribute_count + 1); // tree_idx * instance_count_per_tree * data_row_len
    int instance_data_start_idx = threadIdx.x * (attribute_count + 1) + tree_data_start_idx;

    int *cur_data_line = data + instance_data_start_idx; 
    int *cur_attribute_arr = attribute_arr + blockIdx.x * attribute_count;

    int pos = 0;
    while (!is_leaf(decision_trees[pos])) {
        printf("%s", "should not be in here");
        for (int i = 0; i < attribute_count; i++) {
            if (cur_attribute_arr[i] != decision_trees[pos]) {
                continue;
            }
            pos = cur_data_line + i < 0 ? get_left(pos) : get_right(pos);
        }
    }

    int leaf_class_code = (decision_trees[pos] & (~(1 << 31)));
    leaf_class[thread_pos] = leaf_class_code; 
    leaf_back[thread_pos] = pos;
    // printf("class code: %i  actual_class: %i   class_idx: %i\n", leaf_class_code,
    //        cur_data_line[attribute_count], data_start_idx + attribute_count);

    // TODO test parallel reduction
    if (leaf_class_code == cur_data_line[attribute_count]) {
        atomicAdd(correct_counter, 1);
    }
}

__global__ void counter_increase(int *leaf_counters, int *leaf_classes, int attribute_count) {
    // gridDim: dim3(TREE_COUNT, INSTANCE_COUNT_PER_TREE)
    // blockDim: ATTRIBUTE_COUNT_PER_TREE * 2 (for binary attributes) * CLASS_COUNT

    // input: an array of leaf_counters and leaf_classes reached from tree_traversal

    // Each leaf counter is represented by a block and uses one thread for each attribute i and
    // value j.
    // Row 0 stores the total number of times value n_ij appeared.
    // Row 1 is a mask that keeps track of which attributes have been already used in internal nodes
    // along the path.
    // Row 2 and onwards stores partial counters n_ijk for each class k.

    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int counter_start_pos = block_id * (blockDim.x + 2 * attribute_count);

    int leaf_class = leaf_classes[block_id]; 
    int k = threadIdx.x / (attribute_count * 2); // row

    if (leaf_class != k) return;

    int col = threadIdx.x % (attribute_count * 2);

    int n_ij_idx = counter_start_pos + col; // first row
    int mask_idx = counter_start_pos + attribute_count * 2 + col; // second row
    int n_ijk_idx = counter_start_pos + attribute_count * 2 * k + threadIdx.x;

    atomicAdd(&leaf_counters[n_ij_idx], leaf_counters[mask_idx]);
    atomicAdd(&leaf_counters[n_ijk_idx], leaf_counters[mask_idx]);
}

__global__ void compute_information_gain(int *leaf_counters, 
        int *info_gain_vals, 
        int class_count) {
    // each leaf_counter is mapped to one block in the 1D grid
    // each block needs as many threads as twice number of the (binary) attributes

    // output: a vector with the attributes information gain  values for all leaves in each of the trees

    // gridDim: dim3(TREE_COUNT, LEAF_COUNT_PER_TREE)
    // blockDim: attributes_per_tree * 2

    int tree_id = blockIdx.x;
    int tree_count = gridDim.x;
    int leaf_id = blockIdx.y;
    int leaf_count = gridDim.y;

    int block_id = blockIdx.x + blockIdx.y * gridDim.x; 
    int thread_pos = threadIdx.x + block_id * blockDim.x;

    int *cur_leaf_counter_col = leaf_counters + thread_pos; // TODO

    int a_ij = cur_leaf_counter_col[0];
    int sum = 0;

    for (int i = 0; i < class_count; i++) {
        int a_ijk = cur_leaf_counter_col[2 + i];

        float param = a_ijk / a_ij; // TODO float division by zero returns INF
        asm("max.f32 %0, %1, %2;" : "=f"(param) : "f"(param), "f"((float) 0.0));
        sum += -(param) * log(param);
    }

    info_gain_vals[thread_pos] = -sum;

    __syncthreads();

    int i_00 = 0, i_01 = 0, i_idx = 0;

    if (threadIdx.x % 2 == 0) {
        i_00 = info_gain_vals[thread_pos];
        i_01 = info_gain_vals[thread_pos + 1];
        i_idx = (threadIdx.x << 1) + block_id * blockDim.x;
    }

    __syncthreads();

    if (threadIdx.x % 2 == 0) {
        info_gain_vals[i_idx] = i_00 + i_01;
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
    printf("confidence is: %g\n", confidence);
    return sqrt(((range * range) * log(1.0 / confidence)) / (2.0 * n));
}

__global__ void node_split(float *info_gain_vals, 
        int *attribute_arr, 
        unsigned int *node_split_decisions, 
        int attribute_count,
        int r,
        int delta,
        int n_min) {
    // <<<TREE_COUNT, LEAF_COUNT_PER_TREE>>>
    // note: different from paper by using one thread per leaf
    // output: an array of decisions whether a leaf needs to be split

    int thread_pos = threadIdx.x + blockIdx.x * blockDim.x;

    thrust::sort_by_key(thrust::seq, 
            info_gain_vals + thread_pos, 
            info_gain_vals + thread_pos + attribute_count,
            attribute_arr);

    float first_best = info_gain_vals[thread_pos + attribute_count - 1];
    float second_best = info_gain_vals[thread_pos + attribute_count - 2];

    float hoeffding_bound = compute_hoeffding_bound(r, delta, n_min);

    unsigned int decision = 0;
    if (first_best - second_best > hoeffding_bound) {
        // split
        decision |= (1 << 31);
        decision |= attribute_arr[thread_pos];
    }

    node_split_decisions[thread_pos] = decision;
}

int main(void) {
    const int TREE_COUNT = 1;
    cout << "Number of decision trees: " << TREE_COUNT << endl;

    const int INSTANCE_COUNT_PER_TREE = 200;
    cout << "Instance count per tree: " << INSTANCE_COUNT_PER_TREE << endl;

    const float N_MIN = 200;
    const float DELTA = pow((float) 10.0, -7);
    const float R = 1;

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
    int h_attribute_arr[TREE_COUNT * ATTRIBUTE_COUNT_PER_TREE];
    int *d_attribute_arr;
    for (int i = 0; i < TREE_COUNT; i++) {
        select_k_attributes(h_attribute_arr + i * ATTRIBUTE_COUNT_PER_TREE, ATTRIBUTE_COUNT_TOTAL, ATTRIBUTE_COUNT_PER_TREE);
    }

    // init decision tree
    cout << "\nAllocating memory on host..." << endl;
    void *allocated = malloc(NODE_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
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

    if (!allocate_memory_on_device(&d_attribute_arr, "attribute_arr", ATTRIBUTE_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }

    if (!allocate_memory_on_device(&d_leaf_class, "leaf_class", LEAF_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }

    if (!allocate_memory_on_device(&d_leaf_back, "leaf_back", LEAF_COUNT_PER_TREE * TREE_COUNT)) {
        return 1;
    }

    int *d_leaf_counters;
    if (!allocate_memory_on_device(&d_leaf_counters, "leaf_counters", TREE_COUNT *
                LEAF_COUNT_PER_TREE * ATTRIBUTE_COUNT_PER_TREE * 2 * (CLASS_COUNT + 2))) {
        return 1;
    }

    int *d_info_gain_vals;
    if (!allocate_memory_on_device(&d_info_gain_vals, "info_gain_vals", TREE_COUNT *
                LEAF_COUNT_PER_TREE)) {
        return 1;
    }

    cout << "\nInitializing training data arrays..." << endl;

    int *h_data = (int*) malloc(INSTANCE_COUNT_PER_TREE * (ATTRIBUTE_COUNT_PER_TREE + 1) *
            TREE_COUNT * sizeof(int));

    int *d_data;
    if (!allocate_memory_on_device(&d_data, "data", INSTANCE_COUNT_PER_TREE *
                (ATTRIBUTE_COUNT_PER_TREE + 1) * TREE_COUNT)) {
        return 1;
    }

    cout << "cudaMemcpy..." << endl;
    gpuErrchk(cudaMemcpy(d_decision_trees, h_decision_trees, NODE_COUNT_PER_TREE * TREE_COUNT 
                * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_attribute_arr, h_attribute_arr, ATTRIBUTE_COUNT_PER_TREE * TREE_COUNT *
                sizeof(int), cudaMemcpyHostToDevice));

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

        for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
            
            for (int instance_idx = 0; instance_idx < INSTANCE_COUNT_PER_TREE; instance_idx++) {
                if (!getline(file, line)) {
                    eof = true;
                    break;
                }

                raw_data_row = split(line, ",");
                data_start_idx = (ATTRIBUTE_COUNT_PER_TREE + 1) * INSTANCE_COUNT_PER_TREE * tree_idx 
                    + (ATTRIBUTE_COUNT_PER_TREE + 1) * instance_idx;

                for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
                    // int val = strtod(raw_data[cur_attribute_arr[i]].c_str(), NULL) < 0 ? -1 : 1; // activity-recognition data
                    int attribute_offset = ATTRIBUTE_COUNT_PER_TREE * tree_idx + i;
                    int val = strcmp(raw_data_row[h_attribute_arr[attribute_offset]].c_str(), (const char*) "value1") == 0 ? -1 : 1;

                    h_data[data_start_idx + i] = val;
                }

                h_data[data_start_idx + ATTRIBUTE_COUNT_PER_TREE] = class_code_map[raw_data_row[raw_data_row.size() - 1]]; // class
                // cout << "h_data_class_idx: " << data_start_idx + ATTRIBUTE_COUNT_PER_TREE << endl;
                // cout << "h_data class is: " << h_data[data_start_idx + ATTRIBUTE_COUNT_PER_TREE] << endl;
            }

            if (eof) {
                break; // TODO
            }
        }

        gpuErrchk(cudaMemcpy((void *) d_data, (void *) h_data, INSTANCE_COUNT_PER_TREE *
                    (ATTRIBUTE_COUNT_PER_TREE + 1) * sizeof(int), cudaMemcpyHostToDevice));

        cout << "\nlaunching tree_traversal kernel..." << endl;

        block_count = TREE_COUNT;
        thread_count = INSTANCE_COUNT_PER_TREE;

        cudaMemset(d_correct_counter, 0, sizeof(int));

        cout << "launching " << block_count * thread_count << " threads for tree_traversal" << endl;

        tree_traversal<<<block_count, thread_count>>>(d_decision_trees,
                d_attribute_arr,
                d_data,
                d_leaf_class,
                d_leaf_back,
                d_correct_counter,
                ATTRIBUTE_COUNT_PER_TREE);

        cudaDeviceSynchronize();
        cudaGetLastError();

        cout << "tree_traversal completed" << endl;

        gpuErrchk(cudaMemcpy(&h_correct_counter, d_correct_counter, sizeof(int), cudaMemcpyDeviceToHost));
        cout << "h_correct_counter: " << h_correct_counter << endl;
        double accuracy = (double) h_correct_counter / (INSTANCE_COUNT_PER_TREE * TREE_COUNT);
        cout << INSTANCE_COUNT_PER_TREE * TREE_COUNT << ": " << accuracy << endl;

        cout << "\nlaunching counter_increase kernel..." << endl;

        block_count = LEAF_COUNT_PER_TREE;
        thread_count = ATTRIBUTE_COUNT_PER_TREE * 2;
        // counter_increase<<<block_count, thread_count>>>();

        cout << "counter_increase completed" << endl;

        cout << "\nlanuching compute_information_gain kernel..." << endl;

        dim3 grid(TREE_COUNT, LEAF_COUNT_PER_TREE);
        thread_count = ATTRIBUTE_COUNT_PER_TREE * 2;
        compute_information_gain<<<grid, thread_count>>>(d_leaf_counters,
                d_info_gain_vals,
                CLASS_COUNT);

        cout << "compute_information_gain completed" << endl;

        cout << "\nlaunching node_split kernel..." << endl;

        // node_split<<<TREE_COUNT, LEAF_COUNT_PER_TREE>>>(); 

        cout << "node_split completed" << endl;

        break; // TODO
    }

    cudaFree(d_decision_trees);
    cudaFree(d_attribute_arr);
    cudaFree(d_leaf_class);
    cudaFree(d_leaf_back);
    cudaFree(d_leaf_counters);
    cudaFree(d_data);
    cudaFree(d_info_gain_vals);

    return 0;
}
