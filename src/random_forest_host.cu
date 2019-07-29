#include "random_forest_host.cuh"
#include "common.h"

using namespace std;

extern "C" void setup_kernel_host(curandState* d_state) {
    setup_kernel<<<GROWING_TREE_COUNT, INSTANCE_COUNT_PER_TREE>>>(d_state);
    gpuErrchk(cudaDeviceSynchronize());
}

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
        ofstream& log_file) {

    gpuErrchk(cudaMemcpy((void *) d_data, (void *) h_data, data_len * sizeof(int),
                cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy((void *) d_class_count_arr, (void *) class_count_arr, CLASS_COUNT
                * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_correct_counter, 0, sizeof(int)));
    gpuErrchk(cudaMemset(d_tree_error_count, 0, TREE_COUNT * sizeof(int)));
    gpuErrchk(cudaMemset(d_confusion_matrix, 0, CLASS_COUNT * CLASS_COUNT * sizeof(int)));
    gpuErrchk(cudaMemset(d_forest.tree_confusion_matrices, 0, TREE_COUNT * CLASS_COUNT * CLASS_COUNT
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
            d_forest.decision_trees,
            d_forest.leaf_class,
            d_forest.samples_seen_count,
            d_forest.tree_confusion_matrices,
            d_tree_active_status,
            d_data,
            d_reached_leaf_ids,
            d_is_leaf_active,
            d_correct_counter,
            d_forest_vote,
            d_forest_vote_idx_arr,
            d_weights,
            d_tree_error_count,
            d_confusion_matrix,
            d_class_count_arr,
            majority_class,
            NODE_COUNT_PER_TREE,
            LEAF_COUNT_PER_TREE,
            ATTRIBUTE_COUNT_TOTAL,
            CLASS_COUNT,
            d_state);

    gpuErrchk(cudaDeviceSynchronize());

}

extern "C" void counter_increase_host(
        forest_t& d_forest,
        int* d_tree_active_status,
        int* d_reached_leaf_ids,
        int* d_data,
        int* d_weights) {

    counter_increase
        <<<dim3(GROWING_TREE_COUNT, INSTANCE_COUNT_PER_TREE), ATTRIBUTE_COUNT_TOTAL>>>(
                d_forest.leaf_counters,
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
        forest_t& d_forest,
        int* d_is_leaf_active,
        int* h_tree_active_status,
        int* d_tree_active_status,
        float* d_info_gain_vals,
        int* h_attribute_val_arr,
        int* d_attribute_val_arr) {

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
            d_forest.leaf_counters,
            d_forest.leaf_class,
            d_is_leaf_active,
            d_tree_active_status,
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
        int* d_is_leaf_active,
        int* d_leaf_back,
        int* d_tree_active_status,
        int* d_attribute_val_arr,
        int* h_attribute_idx_arr,
        int* d_attribute_idx_arr,
        int* d_node_split_decisions,
        int* d_samples_seen_count) {

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
        forest_t& d_forest,
        int* d_is_leaf_active,
        int* d_tree_active_status,
        int* d_node_split_decisions,
        int* d_attribute_val_arr) {

    node_split<<<GROWING_TREE_COUNT, LEAF_COUNT_PER_TREE>>>(
            d_forest.decision_trees,
            d_forest.leaf_counters,
            d_forest.leaf_class,
            d_forest.leaf_back,
            d_forest.leaf_id_range_end,
            d_forest.samples_seen_count,
            d_is_leaf_active,
            d_tree_active_status,
            d_node_split_decisions,
            d_attribute_val_arr,
            LEAF_COUNTER_SIZE,
            NODE_COUNT_PER_TREE,
            LEAF_COUNT_PER_TREE,
            ATTRIBUTE_COUNT_PER_TREE,
            ATTRIBUTE_COUNT_TOTAL,
            CLASS_COUNT);

    gpuErrchk(cudaDeviceSynchronize());
}

extern "C" void reset_tree_host(
        forest_t d_forest,
        int* h_reset_tree_idx_arr,
        int* d_reset_tree_idx_arr,
        int reset_tree_count) {

    gpuErrchk(cudaMemcpy(d_reset_tree_idx_arr, h_reset_tree_idx_arr,
                reset_tree_count * sizeof(int), cudaMemcpyHostToDevice));

    reset_tree<<<1, reset_tree_count>>>(
            d_reset_tree_idx_arr,
            d_forest.decision_trees,
            d_forest.leaf_counters,
            d_forest.leaf_class,
            d_forest.leaf_back,
            d_forest.leaf_id_range_end,
            d_forest.samples_seen_count,
            d_forest.tree_confusion_matrices,
            NODE_COUNT_PER_TREE,
            LEAF_COUNT_PER_TREE,
            LEAF_COUNTER_SIZE,
            ATTRIBUTE_COUNT_TOTAL,
            CLASS_COUNT);

    gpuErrchk(cudaDeviceSynchronize());
}
