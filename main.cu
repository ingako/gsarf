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

#include "ADWIN.cu"
#include "LRU_state.cu"

#include "common.h"
#include "random_forest_host.cu"
#include "drift_detection.cu"

using namespace std;

bool STATE_ADAPTIVE;

int TREE_COUNT;
int TREE_DEPTH_PARAM;
int INSTANCE_COUNT_PER_TREE;
int SAMPLE_FREQUENCY;
int CLASS_COUNT;

int FOREGROUND_TREE_COUNT;
int GROWING_TREE_COUNT;

int ATTRIBUTE_COUNT_TOTAL;
int ATTRIBUTE_COUNT_PER_TREE;
int TREE_DEPTH;

int NODE_COUNT_PER_TREE;
int LEAF_COUNT_PER_TREE;

int LEAF_COUNTER_SIZE;
int LEAF_COUNTERS_SIZE_PER_TREE;
long ALL_LEAF_COUNTERS_SIZE;

float r;
float delta;

double warning_delta;
double drift_delta;

double kappa_threshold;
double bg_tree_add_delta;
int CPU_TREE_POOL_SIZE;

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

bool prepare_data(
        ifstream& data_file,
        int* h_data,
        map<string, int>& class_code_map,
        int* class_count_arr,
        int& majority_class) {

    int h_data_idx = 0;

    string line;
    for (int instance_idx = 0; instance_idx < INSTANCE_COUNT_PER_TREE; instance_idx++) {
        if (!getline(data_file, line)) {
            // reached end of line
            return false;
        }

        vector<string> raw_data_row = split(line, ",");

        for (int i = 0; i < ATTRIBUTE_COUNT_TOTAL; i++) {
            int val = stoi(raw_data_row[i]);
            h_data[h_data_idx++] = val;
        }

        int cur_class_code = class_code_map[raw_data_row[ATTRIBUTE_COUNT_TOTAL]];
        h_data[h_data_idx] = cur_class_code;
        class_count_arr[cur_class_code]++;

        h_data_idx++;
    }

    int majority_class_count = 0;
    for (int i = 0; i < CLASS_COUNT; i++) {
        if (majority_class_count < class_count_arr[i]) {
            majority_class_count = class_count_arr[i];
            majority_class = i;
        }
    }

    return true;
}

void evaluate(
        int* d_correct_counter,
        int* h_confusion_matrix,
        int* d_confusion_matrix,
        double& window_accuracy,
        double& window_kappa,
        int& sample_count_iter,
        int iter_count,
        ofstream& log_file,
        ofstream& output_file) {

    int h_correct_counter = 0;
    gpuErrchk(cudaMemcpy(&h_correct_counter, d_correct_counter, sizeof(int),
                cudaMemcpyDeviceToHost));

    log_file << "h_correct_counter: " << h_correct_counter << endl;

    double accuracy = (double) h_correct_counter / INSTANCE_COUNT_PER_TREE;
    window_accuracy = (sample_count_iter * window_accuracy + accuracy)
        / (sample_count_iter + 1);

    gpuErrchk(cudaMemcpy(h_confusion_matrix, d_confusion_matrix,
                CLASS_COUNT * CLASS_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

    double kappa = get_kappa(h_confusion_matrix, CLASS_COUNT, accuracy,
            INSTANCE_COUNT_PER_TREE);
    window_kappa = (sample_count_iter * window_kappa + kappa) / (sample_count_iter + 1);

    sample_count_iter++;;
    int sample_count_total = sample_count_iter * INSTANCE_COUNT_PER_TREE; // avoid expensive mod

    if (sample_count_total >= SAMPLE_FREQUENCY) {
        output_file << iter_count * INSTANCE_COUNT_PER_TREE
            << "," << window_accuracy * 100
            << "," << window_kappa * 100 << endl;

        sample_count_iter = 0;
        window_accuracy = 0.0;
        window_kappa = 0.0;
    }
}

void tree_memcpy(
        forest_t& from_forest,
        int from_tree_idx,
        forest_t& to_forest,
        int to_tree_idx,
        bool is_background_tree) {

    tree_t from_tree = {
        from_forest.decision_trees + from_tree_idx * NODE_COUNT_PER_TREE,
        from_forest.leaf_class + from_tree_idx * LEAF_COUNT_PER_TREE,
        from_forest.leaf_back + from_tree_idx * LEAF_COUNT_PER_TREE,
        from_forest.leaf_id_range_end + from_tree_idx * LEAF_COUNT_PER_TREE,
        from_forest.leaf_counters + from_tree_idx * LEAF_COUNTERS_SIZE_PER_TREE,
        from_forest.samples_seen_count + from_tree_idx * LEAF_COUNT_PER_TREE
    };

    tree_t to_tree = {
        to_forest.decision_trees + to_tree_idx * NODE_COUNT_PER_TREE,
        to_forest.leaf_class + to_tree_idx * LEAF_COUNT_PER_TREE,
        to_forest.leaf_back + to_tree_idx * LEAF_COUNT_PER_TREE,
        to_forest.leaf_id_range_end + to_tree_idx * LEAF_COUNT_PER_TREE,
        to_forest.leaf_counters + to_tree_idx * LEAF_COUNTERS_SIZE_PER_TREE,
        to_forest.samples_seen_count + to_tree_idx * LEAF_COUNT_PER_TREE
    };

    memcpy(to_tree.tree, from_tree.tree, NODE_COUNT_PER_TREE * sizeof(int));

    memcpy(to_tree.leaf_class, from_tree.leaf_class, LEAF_COUNT_PER_TREE * sizeof(int));

    memcpy(to_tree.leaf_back, from_tree.leaf_back, LEAF_COUNT_PER_TREE * sizeof(int));

    memcpy(to_tree.leaf_id_range_end, from_tree.leaf_id_range_end,
            LEAF_COUNT_PER_TREE * sizeof(int));

    if (is_background_tree) {
        memcpy(to_tree.leaf_counter, from_tree.leaf_counter,
                LEAF_COUNTERS_SIZE_PER_TREE * sizeof(int));

        memcpy(to_tree.samples_seen_count, from_tree.samples_seen_count,
                LEAF_COUNT_PER_TREE * sizeof(int));
    }
}

void forest_data_transfer(forest_t& from_forest, forest_t& to_forest, cudaMemcpyKind direction) {

    gpuErrchk(cudaMemcpy(from_forest.decision_trees, to_forest.decision_trees, TREE_COUNT
                * NODE_COUNT_PER_TREE * sizeof(int), direction));

    gpuErrchk(cudaMemcpy(from_forest.leaf_class, to_forest.leaf_class, TREE_COUNT
                * LEAF_COUNT_PER_TREE * sizeof(int), direction));

    gpuErrchk(cudaMemcpy(from_forest.leaf_back, to_forest.leaf_back, TREE_COUNT
                * LEAF_COUNT_PER_TREE * sizeof(int), direction));

    gpuErrchk(cudaMemcpy(from_forest.leaf_id_range_end, to_forest.leaf_id_range_end, TREE_COUNT
                * LEAF_COUNT_PER_TREE * sizeof(int), direction));

    gpuErrchk(cudaMemcpy(from_forest.leaf_counters, to_forest.leaf_counters,
                ALL_LEAF_COUNTERS_SIZE * sizeof(int), direction));

    gpuErrchk(cudaMemcpy(from_forest.samples_seen_count, to_forest.samples_seen_count,
                GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE * sizeof(int), direction));

    if (direction == cudaMemcpyDeviceToHost) {
        gpuErrchk(cudaMemcpy(from_forest.tree_confusion_matrices, to_forest.tree_confusion_matrices,
                    TREE_COUNT * CLASS_COUNT * CLASS_COUNT * sizeof(int), direction));
    }
}

int main(int argc, char *argv[]) {

    TREE_COUNT = 1;
    TREE_DEPTH_PARAM = -1;
    INSTANCE_COUNT_PER_TREE = 200;
    SAMPLE_FREQUENCY = 1000;

    float n_min = 50; // hoeffding bound parameter, grace_period

    kappa_threshold = 0.1;
    int edit_distance_threshold = 50;
    bg_tree_add_delta = 0.01;

    STATE_ADAPTIVE = false;

    warning_delta = 0.001;
    drift_delta = 0.00001;

    string data_path = "data/covtype";
    string data_file_name = "covtype_binary_attributes.csv";

    int opt;
    while ((opt = getopt(argc, argv, "b:t:i:p:n:s:d:g:k:e:x:y:rc")) != -1) {
        switch (opt) {
	        case 'b':
		        bg_tree_add_delta = atof(optarg);
		        break;
            case 'c':
                STATE_ADAPTIVE = true;
                break;
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
            case 'x':
                warning_delta = atof(optarg);
                break;
            case 'y':
                drift_delta = atof(optarg);
                break;
            case 'r':
                // Use a different seed value for each run
                srand(time(NULL));
                break;
        }
    }

    FOREGROUND_TREE_COUNT = TREE_COUNT;
    GROWING_TREE_COUNT = TREE_COUNT * 2;
    TREE_COUNT *= 3;

    ofstream log_file;
    log_file.open("log_file.txt");

    log_file << "TREE_COUNT = " << TREE_COUNT << endl
        << "GROWING_TREE_COUNT = " << GROWING_TREE_COUNT << endl
        << "INSTANCE_COUNT_PER_TREE = " << INSTANCE_COUNT_PER_TREE << endl;

    log_file << "edit_distance_threshold: " << edit_distance_threshold << endl
        << "kappa_threshold: " << kappa_threshold << endl;

    size_t lastindex = data_file_name.find_last_of(".");
    string output_path = data_path + "/result_gpu_";

    if (!STATE_ADAPTIVE) {
        output_path += "garf_";
    }

    output_path += data_file_name.substr(0, lastindex) + ".csv";

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

    ATTRIBUTE_COUNT_TOTAL = split(line, ",").size() - 1;
    ATTRIBUTE_COUNT_PER_TREE = (int) sqrt(ATTRIBUTE_COUNT_TOTAL) + 1;

    TREE_DEPTH =
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
    CLASS_COUNT = line_count;
    log_file << "CLASS_COUNT = " << CLASS_COUNT << endl;

    // hoeffding bound parameters
    delta = 0.05;
    r = log2(CLASS_COUNT); // range of merit = log2(num_of_classes)

    log_file << endl
        << "hoeffding bound parameters: " << endl
        << "n_min = " << n_min << endl
        << "delta = " << delta << endl
        << "r     = " << r     << endl;


    // init decision tree
    log_file << "\nAllocating memory on host..." << endl;
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
    CPU_TREE_POOL_SIZE = FOREGROUND_TREE_COUNT * 40;
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
    int* tree_id_to_forest_idx = (int*) malloc(CPU_TREE_POOL_SIZE * sizeof(int));

    for (int i = 0; i < FOREGROUND_TREE_COUNT; i++) {
        forest_idx_to_tree_id[i] = i;
        tree_id_to_forest_idx[i] = i;
    }

    for (int i = FOREGROUND_TREE_COUNT; i < TREE_COUNT; i++) {
        forest_idx_to_tree_id[i] = -1;
    }

    for (int i = FOREGROUND_TREE_COUNT; i < CPU_TREE_POOL_SIZE; i++) {
        tree_id_to_forest_idx[i] = -1;
    }

    gpuErrchk(cudaMemcpy(d_decision_trees, h_decision_trees, NODE_COUNT_PER_TREE * TREE_COUNT
                * sizeof(int), cudaMemcpyHostToDevice));


    allocated = malloc(LEAF_COUNT_PER_TREE * CPU_TREE_POOL_SIZE * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for cpu_leaf_class failed" << endl;
        return 1;
    }
    int *cpu_leaf_class = (int*) allocated; // stores the class for a given leaf

    allocated = malloc(LEAF_COUNT_PER_TREE * CPU_TREE_POOL_SIZE * sizeof(int));
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

    allocated = malloc(LEAF_COUNT_PER_TREE * TREE_COUNT * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocation for leaf_id_range failed" << endl;
    }
    int* h_leaf_id_range_end = (int*) allocated;
    for (int i = 0; i < TREE_COUNT; i++) {
        int* cur_leaf_id_range_end = h_leaf_id_range_end + i * LEAF_COUNT_PER_TREE;
        cur_leaf_id_range_end[0] = LEAF_COUNT_PER_TREE - 1;
    }

    int* d_leaf_id_range_end;
    if (!allocate_memory_on_device(&d_leaf_id_range_end, "leaf_id_range_end", LEAF_COUNT_PER_TREE
                * TREE_COUNT)) {
        return 1;
    }
    gpuErrchk(cudaMemcpy(d_leaf_id_range_end, h_leaf_id_range_end, LEAF_COUNT_PER_TREE * TREE_COUNT
                * sizeof(int), cudaMemcpyHostToDevice));


    allocated = malloc(LEAF_COUNT_PER_TREE * CPU_TREE_POOL_SIZE * sizeof(int));
    if (allocated == NULL) {
        log_file << "host error: memory allocate for cpu_leaf_id_range_end failed" << endl;
        return 1;
    }
    int* cpu_leaf_id_range_end = (int*) allocated;


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

    LEAF_COUNTER_SIZE = ATTRIBUTE_COUNT_TOTAL * 2 * (CLASS_COUNT + 2);
    LEAF_COUNTERS_SIZE_PER_TREE = LEAF_COUNT_PER_TREE * LEAF_COUNTER_SIZE;
    ALL_LEAF_COUNTERS_SIZE = TREE_COUNT * LEAF_COUNTERS_SIZE_PER_TREE;

    int* h_leaf_counters = (int*) calloc(ALL_LEAF_COUNTERS_SIZE, sizeof(int));

    long cpu_leaf_counters_size = (long) LEAF_COUNTERS_SIZE_PER_TREE
        * CPU_TREE_POOL_SIZE;
    int* cpu_leaf_counters = (int*) malloc(cpu_leaf_counters_size * sizeof(int));

    // init mask row
    for (int tree_idx = 0; tree_idx < TREE_COUNT; tree_idx++) {
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

    int info_gain_vals_len = GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE * ATTRIBUTE_COUNT_PER_TREE * 2;

#if Debug

    float *h_info_gain_vals = (float*) malloc(info_gain_vals_len * sizeof(float));

#endif


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

    // allocate memory for node_split_decisions
    int *d_node_split_decisions;
    int node_split_decisions_len = LEAF_COUNT_PER_TREE * GROWING_TREE_COUNT;

    if (!allocate_memory_on_device(&d_node_split_decisions, "node_split_decisions",
                node_split_decisions_len)) {
        return 1;
    }

    int samples_seen_count_len = TREE_COUNT * LEAF_COUNT_PER_TREE;
    int *h_samples_seen_count = (int*) calloc(samples_seen_count_len, sizeof(int));
    int *cpu_samples_seen_count = (int*) malloc(LEAF_COUNT_PER_TREE
            * CPU_TREE_POOL_SIZE * sizeof(int));
    int *d_samples_seen_count;
    if (!allocate_memory_on_device(&d_samples_seen_count, "samples_seen_count",
                samples_seen_count_len)) {
        return 1;
    }

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
    if (!allocate_memory_on_device(&d_weights, "weights", GROWING_TREE_COUNT
                * INSTANCE_COUNT_PER_TREE)) {
        return 1;
    }

    // one warning and drift detector per tree to monitor accuracy
    // initialized with the default construct where delta=0.001
    ADWIN* warning_detectors[FOREGROUND_TREE_COUNT];
    ADWIN* drift_detectors[FOREGROUND_TREE_COUNT];

    for (int i = 0; i < FOREGROUND_TREE_COUNT; i++) {
        warning_detectors[i] = new ADWIN(warning_delta);
        drift_detectors[i] = new ADWIN(drift_delta);
    }

    int* h_tree_error_count = (int*) calloc(TREE_COUNT, sizeof(int));
    int* d_tree_error_count;
    if (!allocate_memory_on_device(&d_tree_error_count, "tree_error_count", TREE_COUNT)) {
        return 1;
    }

    int* d_drift_tree_idx_arr;
    if (!allocate_memory_on_device(&d_drift_tree_idx_arr, "drift_tree_idx_arr",
                FOREGROUND_TREE_COUNT)) {
        return 1;
    }

    int* d_warning_tree_idx_arr;
    if (!allocate_memory_on_device(&d_warning_tree_idx_arr, "warning_tree_idx_arr",
                FOREGROUND_TREE_COUNT)) {
        return 1;
    }


    // for swapping background trees when drift is detected
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

    state_queue->enqueue(cur_state);


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

    for (int i = FOREGROUND_TREE_COUNT; i < GROWING_TREE_COUNT; i++) {
        h_tree_active_status[i] = 2;
    }

    for (int i = GROWING_TREE_COUNT; i < TREE_COUNT; i++) {
        h_tree_active_status[i] = 4;
    }

    queue<int> next_empty_forest_idx;
    for (int i = GROWING_TREE_COUNT; i < TREE_COUNT; i++) {
        next_empty_forest_idx.push(i);
    }

    vector<candidate_t> forest_candidate_vec; // candidates in forest

    gpuErrchk(cudaMemcpy(d_tree_active_status, h_tree_active_status,
                TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));


    // for calculating kappa measurements
    int* h_confusion_matrix = (int*) malloc(CLASS_COUNT * CLASS_COUNT * sizeof(int));

    int *d_confusion_matrix;
    if (!allocate_memory_on_device(&d_confusion_matrix, "d_confusion_matrix",
                CLASS_COUNT * CLASS_COUNT)) {
        return 1;
    }

    int *h_tree_confusion_matrix = (int*) malloc(TREE_COUNT * CLASS_COUNT * CLASS_COUNT
            * sizeof(int));
    int *d_tree_confusion_matrix;
    if (!allocate_memory_on_device(&d_tree_confusion_matrix, "d_tree_confusion_matrix",
                TREE_COUNT * CLASS_COUNT * CLASS_COUNT)) {
        return 1;
    }

    forest_t h_forest = {
        h_decision_trees,
        h_leaf_class,
        h_leaf_back,
        h_leaf_id_range_end,
        h_leaf_counters,
        h_samples_seen_count,
        h_tree_confusion_matrix
    };

    forest_t d_forest = {
        d_decision_trees,
        d_leaf_class,
        d_leaf_back,
        d_leaf_id_range_end,
        d_leaf_counters,
        d_samples_seen_count,
        d_tree_confusion_matrix
    };

    forest_t cpu_tree_pool = {
        cpu_decision_trees,
        cpu_leaf_class,
        cpu_leaf_back,
        cpu_leaf_id_range_end,
        cpu_leaf_counters,
        cpu_samples_seen_count
    };


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

    log_file << endl << "=====Training Start=====" << endl;

    int *d_correct_counter;
    gpuErrchk(cudaMalloc((void **) &d_correct_counter, sizeof(int)));

    curandState *d_state;
    cudaMalloc(&d_state, GROWING_TREE_COUNT * INSTANCE_COUNT_PER_TREE * sizeof(curandState));

    setup_kernel<<<GROWING_TREE_COUNT, INSTANCE_COUNT_PER_TREE>>>(d_state);
    gpuErrchk(cudaDeviceSynchronize());

    int iter_count = 1;

    int sample_count_iter = 0;
    double window_accuracy = 0.0;
    double window_kappa = 0.0;

    output_file << "#iteration,accuracy,kappa" << endl;

    int matched_pattern = 0;

    while (true) {

        int majority_class = 0;
        int class_count_arr[CLASS_COUNT] = { 0 };

        if (!prepare_data(data_file, h_data, class_code_map, class_count_arr, majority_class)) {
            // reached end of line
            break;
        }

        log_file << "\n=================iteration " << iter_count << "=================" << endl;

        log_file << "\nlaunching tree_traversal kernel..." << endl;

        tree_traversal_host(
            d_forest,
            h_tree_active_status,
            d_tree_active_status,
            h_data,
            d_data,
            data_len,
            d_reached_leaf_ids,
            d_is_leaf_active,
            d_correct_counter,
            d_forest_vote,
            forest_vote_len,
            h_forest_vote_idx_arr,
            d_forest_vote_idx_arr,
            d_weights,
            d_tree_error_count,
            d_confusion_matrix,
            d_class_count_arr,
            majority_class,
            d_state,
            class_count_arr,
            log_file);

        log_file << "tree_traversal completed" << endl;

        evaluate(
                d_correct_counter,
                h_confusion_matrix,
                d_confusion_matrix,
                window_accuracy,
                window_kappa,
                sample_count_iter,
                iter_count,
                log_file,
                output_file);


#if DEBUG

        gpuErrchk(cudaMemcpy(h_decision_trees, d_decision_trees, TREE_COUNT * NODE_COUNT_PER_TREE *
                    sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaMemcpy(h_leaf_class, d_leaf_class, TREE_COUNT * LEAF_COUNT_PER_TREE *
                    sizeof(int), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaMemcpy((void *) h_samples_seen_count, (void *) d_samples_seen_count,
                    samples_seen_count_len * sizeof(int), cudaMemcpyDeviceToHost));

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

        counter_increase_host(
                d_forest,
                d_tree_active_status,
                d_reached_leaf_ids,
                d_data,
                d_weights);

        log_file << "counter_increase completed" << endl;


        log_file << "\nlanuching compute_information_gain kernel..." << endl;

        compute_information_gain_host(
                d_forest,
                d_is_leaf_active,
                h_tree_active_status,
                d_tree_active_status,
                d_info_gain_vals,
                h_attribute_val_arr,
                d_attribute_val_arr);

        log_file << "compute_information_gain completed" << endl;


        log_file << "\nlaunching compute_node_split_decisions kernel..." << endl;

        compute_node_split_decisions_host(
                d_info_gain_vals,
                d_is_leaf_active,
                d_leaf_back,
                d_tree_active_status,
                d_attribute_val_arr,
                h_attribute_idx_arr,
                d_attribute_idx_arr,
                d_node_split_decisions,
                d_samples_seen_count);

        log_file << "compute_node_split_decisions completed" << endl;


        log_file << "\nlaunching node_split kernel..." << endl;

        node_split_host(
                d_forest,
                d_is_leaf_active,
                d_tree_active_status,
                d_node_split_decisions,
                d_attribute_val_arr);

        log_file << "node_split completed" << endl;

        int warning_tree_count = 0;
        int h_warning_tree_idx_arr[FOREGROUND_TREE_COUNT];

        int drift_tree_count = 0;
        int h_drift_tree_idx_arr[FOREGROUND_TREE_COUNT];

        vector<char> target_state(cur_state);
        vector<int> warning_tree_id_list;
        vector<int> drift_tree_id_list;

        detect_drifts(
                h_tree_active_status,
                target_state,
                h_tree_error_count,
                d_tree_error_count,
                warning_tree_count,
                h_warning_tree_idx_arr,
                drift_tree_count,
                h_drift_tree_idx_arr,
                warning_tree_id_list,
                drift_tree_id_list,
                forest_idx_to_tree_id,
                warning_detectors,
                drift_detectors);

        if (warning_tree_count > 0) {

            cout << endl
                << "ಠ_ಠ Warning detected at iter_count = " << iter_count << endl;
            cout << "warning tree forest_idx: ";

            int h_warning_tree_bg_idx_arr[FOREGROUND_TREE_COUNT];

            for (int i = 0; i < warning_tree_count; i++) {
                h_warning_tree_bg_idx_arr[i] = h_warning_tree_idx_arr[i] + FOREGROUND_TREE_COUNT;
                cout << h_warning_tree_idx_arr[i] << " ";
            }
            cout << endl;

            // reset background trees
            reset_tree_host(
                    h_warning_tree_bg_idx_arr,
                    d_warning_tree_idx_arr,
                    warning_tree_count,
                    d_decision_trees,
                    d_leaf_counters,
                    d_leaf_class,
                    d_leaf_back,
                    d_leaf_id_range_end,
                    d_samples_seen_count,
                    d_tree_confusion_matrix);
        }

        if (warning_tree_count > 0 || drift_tree_count > 0) {
            forest_data_transfer(h_forest, d_forest, cudaMemcpyDeviceToHost);
        }


        if (STATE_ADAPTIVE && warning_tree_count > 0) {
            vector<char> closest_state = state_queue->get_closest_state(target_state);

#if Debug
            string target_state_str(target_state.begin(), target_state.end());
            string closest_state_str(closest_state.begin(), closest_state.end());

            cout << "target_state: " << target_state_str << endl;
            cout << "get_closest_state: " << closest_state_str << endl;
#endif


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
                        if (next_empty_forest_idx.empty()) {

                            candidate_t lru_candidate = forest_candidate_vec[0];
                            next_avail_forest_idx = lru_candidate.forest_idx;
                            tree_id_to_forest_idx[lru_candidate.tree_id] = -1;

                            forest_candidate_vec.erase(forest_candidate_vec.begin());

                        } else {
                            next_avail_forest_idx = next_empty_forest_idx.front();
                            next_empty_forest_idx.pop();
                        }

                        if (next_avail_forest_idx < GROWING_TREE_COUNT
                                || next_avail_forest_idx >= TREE_COUNT) {

                            cout << "next_avail_forest_idx out of bound: " <<
                                next_avail_forest_idx << endl;
                            return false;
                        }

                        candidate_t candidate = candidate_t(i, next_avail_forest_idx);
                        forest_candidate_vec.push_back(candidate);

                        tree_memcpy(cpu_tree_pool, i, h_forest, next_avail_forest_idx,
                                false);

                        h_tree_active_status[next_avail_forest_idx] = 5;
                        forest_idx_to_tree_id[next_avail_forest_idx] = i;
                        tree_id_to_forest_idx[i] = next_avail_forest_idx;

                        // reset candiate stats, so kappa becomes 0
                        // h_tree_error_count[next_avail_forest_idx] = INSTANCE_COUNT_PER_TREE;

                        int* candidate_confusion_matrix = h_tree_confusion_matrix
                            + next_avail_forest_idx * CLASS_COUNT * CLASS_COUNT;

                        memset(candidate_confusion_matrix, 0, CLASS_COUNT * CLASS_COUNT * sizeof(int));

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

            for (int i = 0; i < forest_candidate_vec.size(); i++) {
                candidate_t candidate = forest_candidate_vec[i];

                double accuracy = (INSTANCE_COUNT_PER_TREE
                        - h_tree_error_count[candidate.forest_idx])
                    / (double) INSTANCE_COUNT_PER_TREE;

                int* cur_confusion_matrix = h_tree_confusion_matrix + candidate.forest_idx
                    * CLASS_COUNT * CLASS_COUNT;

                forest_candidate_vec[i].kappa = get_kappa(
                        cur_confusion_matrix,
                        CLASS_COUNT,
                        accuracy,
                        INSTANCE_COUNT_PER_TREE);
            }

            sort(forest_candidate_vec.begin(), forest_candidate_vec.end());

            vector<char> next_state(cur_state);

            for (int i = 0; i < drift_tree_id_list.size(); i++) {
                if (cur_tree_pool_size >= CPU_TREE_POOL_SIZE) {
                    // TODO
                    cout << "reached CPU_TREE_POOL_SIZE limit!" << endl;
                    return false;
                }

                int tree_id = drift_tree_id_list[i];
                int forest_tree_idx = h_drift_tree_idx_arr[i];
                int forest_bg_tree_idx = forest_tree_idx + FOREGROUND_TREE_COUNT;

#if DEBUG

                if (tree_id < 0 || tree_id >= CPU_TREE_POOL_SIZE) {
                    cout << "wrong tree_id" << endl;
                    return false;
                }

                if (forest_tree_idx < 0 || forest_tree_idx >= FOREGROUND_TREE_COUNT) {
                    cout << "wrong forest_tree_idx" << endl;
                    return false;
                }

                if (forest_bg_tree_idx < FOREGROUND_TREE_COUNT || forest_bg_tree_idx >=
                        GROWING_TREE_COUNT) {
                    cout << "wrong forest_bg_tree_idx" << endl;
                    return false;
                }

#endif

                double fg_tree_accuracy = (INSTANCE_COUNT_PER_TREE
                        - h_tree_error_count[forest_tree_idx])
                    / (double) INSTANCE_COUNT_PER_TREE;

                int* cur_confusion_matrix = h_tree_confusion_matrix + forest_tree_idx
                    * CLASS_COUNT * CLASS_COUNT;

                double drift_tree_kappa = get_kappa(
                        cur_confusion_matrix,
                        CLASS_COUNT,
                        fg_tree_accuracy,
                        INSTANCE_COUNT_PER_TREE);

                int forest_swap_tree_idx = forest_tree_idx;

                if (STATE_ADAPTIVE && forest_candidate_vec.size() > 0) {

                    candidate_t best_candidate =
                        forest_candidate_vec[forest_candidate_vec.size() - 1];

                    if (best_candidate.kappa - drift_tree_kappa >= kappa_threshold) {
                        forest_swap_tree_idx = best_candidate.tree_id;

#if DEBUG
                        cout << "best_candiate.tree_id: " << best_candidate.tree_id << endl;
                        int* cur_tree = cpu_tree_pool.decision_trees + best_candidate.tree_id *
                            NODE_COUNT_PER_TREE;
                        for (int node_idx = 0; node_idx < NODE_COUNT_PER_TREE; node_idx++) {
                            cout << cur_tree[node_idx] << ",";
                        }
                        cout << endl;
#endif
                    }
                }


                bool add_bg_tree = false;
                if (forest_swap_tree_idx == forest_tree_idx
                        && h_tree_active_status[forest_bg_tree_idx] == 3) {

                    double bg_tree_accuracy = (INSTANCE_COUNT_PER_TREE
                            - h_tree_error_count[forest_bg_tree_idx])
                        / (double) INSTANCE_COUNT_PER_TREE;

                    int* cur_bg_tree_confusion_matrix = h_tree_confusion_matrix
                        + forest_bg_tree_idx * CLASS_COUNT * CLASS_COUNT;

                    double bg_tree_kappa = get_kappa(
                            cur_bg_tree_confusion_matrix,
                            CLASS_COUNT,
                            bg_tree_accuracy,
                            INSTANCE_COUNT_PER_TREE);

                    forest_swap_tree_idx = -1;
                    if (fabs(bg_tree_kappa - drift_tree_kappa) > bg_tree_add_delta) {
                        add_bg_tree = true;
                    }
                }
                h_tree_active_status[forest_bg_tree_idx] = 2;


                if (forest_swap_tree_idx == forest_tree_idx) {
                    continue;
                }


                // put drift tree back to cpu tree pool
                tree_memcpy(h_forest, forest_tree_idx, cpu_tree_pool, tree_id, true);

                tree_id_to_forest_idx[tree_id] = -1;

                if (forest_swap_tree_idx == -1) {

                    // replace drift tree with its background tree
                    tree_memcpy(h_forest, forest_bg_tree_idx, h_forest, forest_tree_idx, true);

                    if (add_bg_tree) {
                        // add background tree to cpu_tree_pool
                        int new_tree_id = cur_tree_pool_size;
                        tree_memcpy(h_forest, forest_bg_tree_idx, cpu_tree_pool, new_tree_id, true);
                        forest_idx_to_tree_id[forest_tree_idx] = new_tree_id;
                        tree_id_to_forest_idx[new_tree_id] = forest_tree_idx;

                        next_state[new_tree_id] = '1';
                        next_state[tree_id] = '0';

                        cur_tree_pool_size++;

                        cur_state = next_state;
                        state_queue->enqueue(cur_state);
                        state_queue->to_string();
                    }


                } else {

                    // find the best candidate and replace it with drift tree
                    if (forest_candidate_vec.empty()) {
                        cout << "forest_candidate_vec should not be empty" << endl;
                        return false;
                    }

                    candidate_t best_candidate = forest_candidate_vec.back();

                    cout << "------------picked candidate tree: "
                        << best_candidate.tree_id << endl;

#if DEBUG

                    if (best_candidate.tree_id < 0
                            || best_candidate.tree_id >= CPU_TREE_POOL_SIZE) {
                        cout << "incorrect best_candidate.tree_id" << endl;
                        return false;
                    }

                    if (best_candidate.forest_idx < GROWING_TREE_COUNT
                            || best_candidate.forest_idx >= TREE_COUNT) {
                        cout << "incorrect best_candidate.forest_idx" << endl;
                        return false;
                    }

#endif

                    // replace drift tree with its candidate tree
                    tree_memcpy(cpu_tree_pool, best_candidate.tree_id, h_forest,
                            forest_tree_idx, true);

                    tree_id_to_forest_idx[best_candidate.tree_id] = forest_tree_idx;
                    forest_idx_to_tree_id[forest_tree_idx] = best_candidate.tree_id;

                    forest_idx_to_tree_id[best_candidate.forest_idx] = -1;

                    next_empty_forest_idx.push(best_candidate.forest_idx);
                    h_tree_active_status[best_candidate.forest_idx] = 4;

                    next_state[best_candidate.tree_id] = '1';
                    next_state[tree_id] = '0';


                    forest_candidate_vec.pop_back();

                    cur_state = next_state;
                    state_queue->enqueue(cur_state);
                    state_queue->to_string();
                }
            }

            cout << "cur_tree_pool_size: " << cur_tree_pool_size << endl;

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

        }


        if (warning_tree_count > 0 || drift_tree_count > 0) {
            forest_data_transfer(d_forest, h_forest,  cudaMemcpyHostToDevice);
        }

        if (drift_tree_count > 0) {

            for (int i = 0; i < drift_tree_count; i++) {
                int bg_tree_forest_idx = h_drift_tree_idx_arr[i] + FOREGROUND_TREE_COUNT;
                h_drift_tree_idx_arr[i] = bg_tree_forest_idx;
                h_tree_active_status[bg_tree_forest_idx] = 2;
            }

            reset_tree_host(
                    h_drift_tree_idx_arr,
                    d_drift_tree_idx_arr,
                    drift_tree_count,
                    d_decision_trees,
                    d_leaf_counters,
                    d_leaf_class,
                    d_leaf_back,
                    d_leaf_id_range_end,
                    d_samples_seen_count,
                    d_tree_confusion_matrix);
        }

        iter_count++;
    }

    log_file << "\ntraining completed" << endl;

    if (STATE_ADAPTIVE) {
        log_file << "cur_tree_pool_size: " << cur_tree_pool_size << endl;
        log_file << "pattern matched: " << matched_pattern << endl;

	    std::ofstream tree_pool_size_log;
	    tree_pool_size_log.open(data_path + "log.tree_pool_size", std::ios_base::app);
	    tree_pool_size_log << data_file_name << " " <<  cur_tree_pool_size << endl;
    }

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
    cudaFree(d_attribute_val_arr);
    cudaFree(d_attribute_idx_arr);
    cudaFree(d_confusion_matrix);
    cudaFree(d_tree_confusion_matrix);

    output_file.close();

    return 0;
}
