#ifndef __random_forest_host__
#define __random_forest_host__

#include <fstream>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "common.h"
#include "random_forest.cuh"

using namespace std;

extern "C" void setup_kernel_host(curandState* d_state);

extern "C" void tree_traversal_host(
        forest_t& d_forest,
        int* h_tree_active_status,
        int* d_tree_active_status,
        int* h_data,
        int* d_data,
        int data_len,
        int* d_reached_leaf_ids,
        int* d_is_leaf_active,
        int* d_correct_counter,
        int* d_forest_vote,
        int forest_vote_len,
        int* h_forest_vote_idx_arr,
        int* d_forest_vote_idx_arr,
        int* d_weights,
        int* d_tree_error_count,
        int* d_confusion_matrix,
        int* d_class_count_arr,
        int majority_class,
        curandState* d_state,
        int* class_count_arr,
        ofstream& log_file);

extern "C" void counter_increase_host(
        forest_t& d_forest,
        int* d_tree_active_status,
        int* d_reached_leaf_ids,
        int* d_data,
        int* d_weights);


extern "C" void compute_information_gain_host(
        forest_t& d_forest,
        int* d_is_leaf_active,
        int* h_tree_active_status,
        int* d_tree_active_status,
        float* d_info_gain_vals,
        int* h_attribute_val_arr,
        int* d_attribute_val_arr);


extern "C" void compute_node_split_decisions_host(
        float* d_info_gain_vals,
        int* d_is_leaf_active,
        int* d_leaf_back,
        int* d_tree_active_status,
        int* d_attribute_val_arr,
        int* h_attribute_idx_arr,
        int* d_attribute_idx_arr,
        int* d_node_split_decisions,
        int* d_samples_seen_count);


extern "C" void node_split_host(
        forest_t& d_forest,
        int* d_is_leaf_active,
        int* d_tree_active_status,
        int* d_node_split_decisions,
        int* d_attribute_val_arr);


extern "C" void reset_tree_host(
        forest_t d_forest,
        int* h_reset_tree_idx_arr,
        int* d_reset_tree_idx_arr,
        int reset_tree_count);

#endif
