#ifndef __random_forest__
#define __random_forest__

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "common.h"

__device__ long get_left(int index);
__device__ long get_right(int index);
__device__ int get_rand(int low, int high, curandState *local_state);
__device__ int poisson(float lambda, curandState *local_state);
__device__ float compute_hoeffding_bound(float range, float confidence, float n);

__global__ void setup_kernel(curandState *state);

__global__ void reset_tree(
        int *reseted_tree_idx_arr,
        int *decision_trees,
        int *leaf_counters,
        int *leaf_class,
        int *leaf_back,
        int *leaf_id_range_end,
        int *samples_seen_count,
        int *tree_confusion_matrix,
        int node_count_per_tree,
        int leaf_count_per_tree,
        int leaf_counter_size,
        int attribute_count_total,
        int class_count);

__global__ void tree_traversal(
        int *decision_trees,
        int *leaf_class,
        int *samples_seen_count,
        int *tree_confusion_matrix,
        int *tree_status,
        int *data,
        int *reached_leaf_ids,
        int *is_leaf_active,
        int *correct_counter,
        int *forest_vote,
        int *forest_vote_idx_arr,
        int *weights,
        int *tree_error_count,
        int *confusion_matrix,
        int *class_count_arr,
        int majority_class,
        int node_count_per_tree,
        int leaf_count_per_tree,
        int attribute_count_total,
        int class_count,
        curandState *state);

__global__ void counter_increase(
        int *leaf_counters,
        int *tree_status,
        int *reached_leaf_ids,
        int *data,
        int *weights,
        int class_count,
        int attribute_count_total,
        int leaf_count_per_tree,
        int leaf_counter_size);

__global__ void compute_information_gain(
        int *leaf_counters,
        int *leaf_class,
        int* is_leaf_active,
        int *tree_status,
        float *info_gain_vals,
        int *attribute_val_arr,
        int attribute_count_per_tree,
        int attribute_count_total,
        int leaf_count_per_tree,
        int class_count,
        int leaf_counter_size);

__global__ void compute_node_split_decisions(
        float* info_gain_vals,
        int* is_leaf_active,
        int* leaf_back,
        int* tree_status,
        int* attribute_val_arr,
        int* attribute_idx_arr,
        int* node_split_decisions,
        int attribute_count_per_tree,
        float r,
        float delta,
        int node_count_per_tree,
        int leaf_count_per_tree,
        int* samples_seen_count);

__global__ void node_split(
        int* decision_trees,
        int* leaf_counters,
        int* leaf_class,
        int* leaf_back,
        int* leaf_id_range_end,
        int* samples_seen_count,
        int* is_leaf_active,
        int* tree_status,
        int* node_split_decisions,
        int* attribute_val_arr,
        int counter_size_per_leaf,
        int node_count_per_tree,
        int leaf_count_per_tree,
        int attribute_count_per_tree,
        int attribute_count_total,
        int class_count);

#endif
