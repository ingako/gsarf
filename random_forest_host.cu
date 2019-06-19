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

#include "common.h"
#include "random_forest.cu"

using namespace std;

extern "C" void tree_traversal_host(
        int *d_decision_trees,
        int *h_tree_active_status,
        int *d_tree_active_status,
        int *h_data,
        int *d_data,
        int data_len,
        int *d_reached_leaf_ids,
        int *d_is_leaf_active,
        int *d_leaf_class,
        int *d_correct_counter,
        int *d_samples_seen_count,
        int *d_forest_vote,
        int forest_vote_len,
        int *h_forest_vote_idx_arr,
        int *d_forest_vote_idx_arr,
        int *d_weights,
        int *d_tree_error_count,
        int tree_error_count_len,
        int *d_confusion_matrix,
        int *d_tree_confusion_matrix,
        int *d_class_count_arr,
        int majority_class,
        int confusion_matrix_size,
        curandState *d_state,
        int *class_count_arr,
        ofstream &log_file) { 

    gpuErrchk(cudaMemcpy((void *) d_data, (void *) h_data, data_len * sizeof(int), 
                cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy((void *) d_class_count_arr, (void *) class_count_arr, CLASS_COUNT
                * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_correct_counter, 0, sizeof(int)));
    gpuErrchk(cudaMemset(d_tree_error_count, 0, tree_error_count_len * sizeof(int)));
    gpuErrchk(cudaMemset(d_confusion_matrix, 0, confusion_matrix_size * sizeof(int)));
    gpuErrchk(cudaMemset(d_tree_confusion_matrix, 0, TREE_COUNT * confusion_matrix_size
                * sizeof(int)));

    gpuErrchk(cudaMemset(d_is_leaf_active, 0, GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE
                * sizeof(int)));
    gpuErrchk(cudaMemset(d_forest_vote, 0, forest_vote_len * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_forest_vote_idx_arr, h_forest_vote_idx_arr, forest_vote_len
                * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_tree_active_status, h_tree_active_status,
                TREE_COUNT * sizeof(int), cudaMemcpyHostToDevice));

    int block_count = TREE_COUNT;
    int thread_count = INSTANCE_COUNT_PER_TREE;

    log_file << "launching " << block_count * thread_count << " threads for tree_traversal" << endl;

    gpuErrchk(cudaDeviceSynchronize());

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

    gpuErrchk(cudaDeviceSynchronize());

}

extern "C" void counter_increase_host(
        int *d_leaf_counters,
        int *d_tree_active_status,
        int *d_reached_leaf_ids,
        int *d_data,
        int *d_weights) {

    counter_increase
        <<<dim3(GROWING_TREE_COUNT, INSTANCE_COUNT_PER_TREE), ATTRIBUTE_COUNT_TOTAL>>>(
                d_leaf_counters,
                d_tree_active_status,
                d_reached_leaf_ids,
                d_data,
                d_weights,
                CLASS_COUNT,
                ATTRIBUTE_COUNT_TOTAL,
                LEAF_COUNT_PER_TREE,
                LEAF_COUNTER_SIZE);

    gpuErrchk(cudaDeviceSynchronize());
}

extern "C" void compute_information_gain_host(
        int *d_leaf_counters,
        int *d_is_leaf_active,
        int *h_tree_active_status,
        int *d_tree_active_status,
        int *d_leaf_class,
        float *d_info_gain_vals,
        int *h_attribute_val_arr,
        int *d_attribute_val_arr) {

    // select k random attributes for each tree
    for (int tree_idx = 0; tree_idx < GROWING_TREE_COUNT; tree_idx++) {

        // select random attributes for active trees only
        if (h_tree_active_status[tree_idx] == 0 || h_tree_active_status[tree_idx] == 2) {
            continue;
        }

        int *cur_attribute_val_arr = h_attribute_val_arr + tree_idx * ATTRIBUTE_COUNT_PER_TREE;

        // choose with replacement increases uncorrelation among trees
        for (int i = 0; i < ATTRIBUTE_COUNT_PER_TREE; i++) {
            cur_attribute_val_arr[i] = rand() % ATTRIBUTE_COUNT_TOTAL;
        }
    }

    gpuErrchk(cudaMemcpy(d_attribute_val_arr, h_attribute_val_arr,
                GROWING_TREE_COUNT * ATTRIBUTE_COUNT_PER_TREE * sizeof(int),
                cudaMemcpyHostToDevice));

    // for sorting information gain array
    // gpuErrchk(cudaMemcpy(d_attribute_idx_arr, h_attribute_idx_arr, attribute_idx_arr_len * sizeof(int), cudaMemcpyHostToDevice));


    dim3 grid(GROWING_TREE_COUNT, LEAF_COUNT_PER_TREE);
    int thread_count = ATTRIBUTE_COUNT_PER_TREE * 2;

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

    gpuErrchk(cudaDeviceSynchronize());

}

extern "C" void compute_node_split_decisions_host(
        float* d_info_gain_vals,
        int *d_is_leaf_active,
        int *d_leaf_back,
        int *d_tree_active_status,
        int *d_attribute_val_arr,
        int *h_attribute_idx_arr,
        int *d_attribute_idx_arr,
        int *d_node_split_decisions,
        int *d_samples_seen_count) {

    gpuErrchk(cudaMemcpy(d_attribute_idx_arr, h_attribute_idx_arr,
                GROWING_TREE_COUNT * LEAF_COUNT_PER_TREE * ATTRIBUTE_COUNT_PER_TREE
                * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_node_split_decisions, 0,
                LEAF_COUNT_PER_TREE * GROWING_TREE_COUNT * sizeof(int)));

    compute_node_split_decisions<<<GROWING_TREE_COUNT, LEAF_COUNT_PER_TREE>>>(
            d_info_gain_vals,
            d_is_leaf_active,
            d_leaf_back,
            d_tree_active_status,
            d_attribute_val_arr,
            d_attribute_idx_arr,
            d_node_split_decisions,
            ATTRIBUTE_COUNT_PER_TREE,
            r,
            delta,
            NODE_COUNT_PER_TREE,
            LEAF_COUNT_PER_TREE,
            d_samples_seen_count);

    gpuErrchk(cudaDeviceSynchronize());

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
}


extern "C" void node_split_host(
        int *d_decision_trees,
        int *d_is_leaf_active,
        int *d_tree_active_status,
        int *d_node_split_decisions,
        int *d_leaf_counters,
        int *d_leaf_class,
        int *d_leaf_back,
        int *d_leaf_id_range_end,
        int *d_attribute_val_arr,
        int *d_samples_seen_count,
        int *d_cur_node_count_per_tree,
        int *d_cur_leaf_count_per_tree) {

    node_split<<<GROWING_TREE_COUNT, LEAF_COUNT_PER_TREE>>>(
            d_decision_trees,
            d_is_leaf_active,
            d_tree_active_status,
            d_node_split_decisions,
            d_leaf_counters,
            d_leaf_class,
            d_leaf_back,
            d_leaf_id_range_end,
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

    gpuErrchk(cudaDeviceSynchronize());
}
