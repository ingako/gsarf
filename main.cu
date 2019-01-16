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

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
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
        int attribute_count) {
    int pos = 0;
    int thread_pos = threadIdx.x + blockIdx.x * blockDim.x;
    int data_start_pos = data[thread_pos];

    while (!is_leaf(decision_trees[pos])) {
        for (int i = 0; i < attribute_count; i++) {
            if (attribute_arr[i] != decision_trees[pos]) {
                continue;
            }
            pos = data[data_start_pos + i] < 0 ? get_left(pos) : get_right(pos);
        }
    }

    leaf_class[thread_pos] = (decision_trees[pos] & (~(1 << 31)));
    leaf_back[thread_pos] = pos;
}

__global__ void counter_increase(int *leaf_counters) {
    // TODO
}

__global__ void compute_information_gain(int *leaf_counters, 
        int *info_gain_vals, 
        int class_count) {
    // each leaf_counter is mapped to one block in the 1D grid
    // each block needs as many threads as twice number of the (binary) attributes
    // output: a vector with the attributes information gain  values for all leaves in each of the trees

    // gridDim: dim3(forest_size, leaf_count)
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

int main(void) {
    const int FOREST_SIZE = 1;
    cout << "Forest size: " << FOREST_SIZE << endl;

    const int INSTANCE_COUNT_PER_TREE = 1;
    cout << "Instance count per tree: " << INSTANCE_COUNT_PER_TREE << endl;
    
    // Use a different seed value for each run
    // srand(time(NULL));
    
    cout << "Reading class file..." << endl; 
    ifstream class_file("data/activity_labels.txt");
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
    cout << "Number of class: " << CLASS_COUNT << endl;

    // prepare attributes
    std::ifstream file("data/train.csv");
    string line;

    getline(file, line);
    const int ATTRIBUTE_COUNT_TOTAL = split_attributes(line, ',').size() - 2; 
    const int ATTRIBUTE_COUNT_PER_TREE = (int) sqrt(ATTRIBUTE_COUNT_TOTAL);

    cout << "Attribute count total: " << ATTRIBUTE_COUNT_TOTAL << endl;
    cout << "Attribute count per tree: " << ATTRIBUTE_COUNT_PER_TREE << endl;

    const unsigned int TREE_NODE_COUNT = (1 << ATTRIBUTE_COUNT_PER_TREE);
    const unsigned int LEAF_COUNT = (TREE_NODE_COUNT >> 1);

    cout << "TREE_NODE_COUNT: " << TREE_NODE_COUNT << " bytes" << endl;
    cout << "LEAF_COUNT: " << LEAF_COUNT << " bytes" << endl;

    // select k random attributes for each tree
    int h_attribute_arr[FOREST_SIZE][ATTRIBUTE_COUNT_PER_TREE];
    for (int i = 0; i < FOREST_SIZE; i++) {
        select_k_attributes(h_attribute_arr[i], ATTRIBUTE_COUNT_TOTAL, ATTRIBUTE_COUNT_PER_TREE);
    }

    // init decision tree
    void *allocated = malloc(TREE_NODE_COUNT * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for decision trees failed" << endl;
        return 1;
    }
    int *h_decision_trees = (int*) allocated;
    int *d_decision_trees;

    allocated = malloc(LEAF_COUNT * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for leaf_class failed" << endl;
        return 1;
    }
    int *h_leaf_class = (int*) allocated; // stores the class for a given leaf
    int *d_leaf_class;

    allocated = malloc(LEAF_COUNT * sizeof(int));
    if (allocated == NULL) {
        cout << "host error: memory allocation for leaf_back failed" << endl;
        return 1;
    }
    int *h_leaf_back = (int*) allocated; // reverse pointer to map a leaf id to an offset in the tree array
    int *d_leaf_back;

    // int h_leaf_counters[(2 + CLASS_COUNT) * ATTRIBUTE_COUNT_PER_TREE * 2 *
    //    LEAF_COUNT * FOREST_SIZE];
    int *d_leaf_counters;
    
    cout << "Init: set root as leaf for each tree in the forest..." << endl;
    for (int i = 0; i < FOREST_SIZE; i++) {
        h_decision_trees[i * TREE_NODE_COUNT] = 0;
        h_decision_trees[i * TREE_NODE_COUNT] |= (1 << 31); // init root node
    }

    cout << "Allocating memory on device..." << endl;

    cudaError_t err;

    cout << "Allocating  " << TREE_NODE_COUNT * FOREST_SIZE * sizeof(int) << " bytes for decision trees on device..." << endl;
    err = cudaMalloc((void **) &d_decision_trees, TREE_NODE_COUNT *
            FOREST_SIZE * sizeof(int)); // allocate global memory on the device
    if (err) {
        cout << "error allocating memory for decision_trees on device: " <<
            TREE_NODE_COUNT * FOREST_SIZE << " bytes" << endl;
        return 1;
    } else {
        cout << "device: memory for decision tree allocated successfully" <<
            endl;
    }
    
    cout << "Allocating " << LEAF_COUNT * sizeof(int) << " bytes for leaf_class on device..." << endl;
    err = cudaMalloc((void **) &d_leaf_class, LEAF_COUNT * sizeof(int));
    if (err) {
        cout << "error allocating memory for leaf_class on device" << endl;
        return 1;
    } else {
        cout << "device: memory for leaf_class allocated successfully. " << endl;
    }

    cout << "Allocating " << LEAF_COUNT * sizeof(int) << " bytes for leaf_back on device..." << endl;
    err = cudaMalloc((void **) &d_leaf_back, LEAF_COUNT * sizeof(int));
    if (err) {
        cout << "error allocating memory for leaf_back on device" << endl;
        return 1;
    } else {
        cout << "device: memory for leaf_back allocated successfully." << endl;
    }

    gpuErrchk(cudaMemcpy(d_decision_trees, h_decision_trees, TREE_NODE_COUNT *
                FOREST_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    cout << "Initialize training data arrays..." << endl;
    int tree_idx = 0;
    int instance_idx = 0;

    int *h_data = (int*) malloc(INSTANCE_COUNT_PER_TREE * (ATTRIBUTE_COUNT_PER_TREE + 1) * sizeof(int));
    int *d_data;
    
    err = cudaMalloc((void**) &d_data, INSTANCE_COUNT_PER_TREE * (ATTRIBUTE_COUNT_PER_TREE + 1) * sizeof(int));
    if (err) {
        cout << "error allocating memory for data array on GPU" << endl;
        return 1;
    }

    int *d_cur_attribute_arr;
    vector<string> arr;

    int block_count;
    int thread_count;

    cout << endl << "Start training..." << endl;
    while (getline(file, line)) {
        arr = split(line, ",");

        int *cur_attribute_arr = h_attribute_arr[tree_idx];
        for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
            int val = strtod(arr[cur_attribute_arr[i]].c_str(), NULL) < 0 ? -1 : 1;
            h_data[instance_idx * ATTRIBUTE_COUNT_PER_TREE + i] = val;
        }
        h_data[instance_idx * ATTRIBUTE_COUNT_PER_TREE] = class_code_map[arr[arr.size() - 1]]; // class

        instance_idx++;

        if (instance_idx == INSTANCE_COUNT_PER_TREE) {
            cudaMemcpy((void *) d_data, (void *) &h_data, INSTANCE_COUNT_PER_TREE * (ATTRIBUTE_COUNT_PER_TREE + 1) * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy((void *) d_cur_attribute_arr, (void *) &cur_attribute_arr, ATTRIBUTE_COUNT_PER_TREE * sizeof(int), cudaMemcpyHostToDevice);

            cout << "launching tree_traversal kernel" << endl;
            
            block_count = FOREST_SIZE;
            thread_count = INSTANCE_COUNT_PER_TREE;
            tree_traversal<<<block_count, thread_count>>>(d_decision_trees,
                    d_cur_attribute_arr,
                    d_data,
                    d_leaf_class,
                    d_leaf_back,
                    ATTRIBUTE_COUNT_PER_TREE);

            cout << "tree_traversal completed" << endl;
            instance_idx = 0;

            cout << "launching counter_increase kernel" << endl;

            block_count = LEAF_COUNT;
            thread_count = ATTRIBUTE_COUNT_PER_TREE * 2;
            // counter_increase<<<block_count, thread_count>>>();

            cout << "counter_increase completed" << endl;

            cout << "lanuching compute_information_gain kernel" << endl;
            
            int *d_info_gain_vals;

            cout << "Allocating info_gain_vals..." << endl;
            err = cudaMalloc((void **) &d_info_gain_vals, FOREST_SIZE * LEAF_COUNT * sizeof(float));
            if (err) {
                cout << "error allocating memory for info_gain_vals" <<endl;
                return 1;
            } else {
                cout << "device: memory for info_gain_vals allocated successfully." << endl;
            }

            dim3 grid(FOREST_SIZE, LEAF_COUNT);
            thread_count = ATTRIBUTE_COUNT_PER_TREE * 2;
            compute_information_gain<<<grid, thread_count>>>(d_leaf_counters,
                    d_info_gain_vals,
                    CLASS_COUNT);

            cout << "compute_information_gain completed" << endl;
            
            cudaFree(d_info_gain_vals);
        }

        break; // TODO 
    }

    cudaFree(d_decision_trees);
    cudaFree(d_leaf_class);
    cudaFree(d_leaf_back);
    cudaFree(d_data);
    cudaFree(d_cur_attribute_arr);
    
    return 0;
}
