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

#include "ADWIN.h"
#include "LRU_state.cpp"

#include "common.h"
#include "random_forest_host.cuh"
#include "state_adaptation.cpp"

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

    setup_kernel_host(d_state);

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


        if (!adapt_state(
                    h_forest,
                    d_forest,
                    cpu_tree_pool,
                    warning_detectors,
                    drift_detectors,
                    h_tree_active_status,
                    h_tree_error_count,
                    d_tree_error_count,
                    forest_idx_to_tree_id,
                    tree_id_to_forest_idx,
                    state_queue,
                    cur_state,
                    forest_candidate_vec,
                    next_empty_forest_idx,
                    cur_tree_pool_size,
                    iter_count)) {

            return false;
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
