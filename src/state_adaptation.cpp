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

bool adapt_state(
        forest_t& h_forest,
        forest_t& d_forest,
        forest_t& cpu_tree_pool,
        ADWIN** warning_detectors,
        ADWIN** drift_detectors,
        int* h_tree_active_status,
        int* h_tree_error_count,
        int* d_tree_error_count,
        int* forest_idx_to_tree_id,
        int* tree_id_to_forest_idx,
        LRU_state* state_queue,
        vector<char>& cur_state,
        vector<candidate_t>& forest_candidate_vec,
        queue<int>& next_empty_forest_idx,
        int& cur_tree_pool_size,
        int iter_count) {

    gpuErrchk(cudaMemcpy((void *) h_tree_error_count, (void *) d_tree_error_count,
                TREE_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

    int warning_tree_count = 0;
    int h_warning_tree_idx_arr[FOREGROUND_TREE_COUNT];

    int drift_tree_count = 0;
    int h_drift_tree_idx_arr[FOREGROUND_TREE_COUNT];

    vector<char> target_state(cur_state);
    vector<int> warning_tree_id_list;
    vector<int> drift_tree_id_list;

    // warning/drift detection only on foreground trees
    // if accuracy decreases, reset the tree
    for (int tree_idx = 0; tree_idx < FOREGROUND_TREE_COUNT; tree_idx++) {
        ADWIN *warning_detector = warning_detectors[tree_idx];
        double old_error = warning_detector->getEstimation();
        bool error_change = warning_detector->setInput(h_tree_error_count[tree_idx]);

        if (error_change && old_error > warning_detector->getEstimation()) {
            error_change = false;
        }

        int bg_tree_pos = tree_idx + FOREGROUND_TREE_COUNT;

        // warning detected
        if (error_change) {
            delete warning_detectors[tree_idx];
            warning_detectors[tree_idx] = new ADWIN(warning_delta);

            // grow background tree
            if (h_tree_active_status[bg_tree_pos] == 2) {
                // start growing if never grown
                h_tree_active_status[bg_tree_pos] = 3;
            }

            if (STATE_ADAPTIVE) {
                target_state[forest_idx_to_tree_id[tree_idx]] = '2';
            }
            warning_tree_id_list.push_back(forest_idx_to_tree_id[tree_idx]);

            h_warning_tree_idx_arr[warning_tree_count] = tree_idx;
            warning_tree_count++;
        }

        ADWIN *drift_detector = drift_detectors[tree_idx];
        old_error = drift_detector->getEstimation();
        error_change = drift_detector->setInput(h_tree_error_count[tree_idx]);

        if (error_change && old_error > drift_detector->getEstimation()) {
            // if error is decreasing, do nothing
            error_change = false;
        }

        if (!error_change) {
            continue;
        }

        // drift detected
        delete warning_detectors[tree_idx];
        delete drift_detectors[tree_idx];

        warning_detectors[tree_idx] = new ADWIN(warning_delta);
        drift_detectors[tree_idx] = new ADWIN(drift_delta);

        h_drift_tree_idx_arr[drift_tree_count] = tree_idx;
        drift_tree_count++;

        drift_tree_id_list.push_back(forest_idx_to_tree_id[tree_idx]);
    }


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


        int* d_warning_tree_idx_arr;
        if (!allocate_memory_on_device(&d_warning_tree_idx_arr, "warning_tree_idx_arr",
                    FOREGROUND_TREE_COUNT)) {
            return 1;
        }


        // reset background trees
        reset_tree_host(
                d_forest,
                h_warning_tree_bg_idx_arr,
                d_warning_tree_idx_arr,
                warning_tree_count);
    }

    if (warning_tree_count > 0 || drift_tree_count > 0) {
        forest_data_transfer(h_forest, d_forest, cudaMemcpyDeviceToHost);
    }


    if (STATE_ADAPTIVE && warning_tree_count > 0) {
        vector<char> closest_state = state_queue->get_closest_state(target_state);

#if DEBUG
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

                    int* candidate_confusion_matrix = h_forest.tree_confusion_matrices
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

            int* cur_confusion_matrix = h_forest.tree_confusion_matrices + candidate.forest_idx
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
                // evict LRU tree
                
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

            int* cur_confusion_matrix = h_forest.tree_confusion_matrices + forest_tree_idx
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

                int* cur_bg_tree_confusion_matrix = h_forest.tree_confusion_matrices
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

        int* d_drift_tree_idx_arr;
        if (!allocate_memory_on_device(&d_drift_tree_idx_arr, "drift_tree_idx_arr",
                    FOREGROUND_TREE_COUNT)) {
            return 1;
        }

        reset_tree_host(
                d_forest,
                h_drift_tree_idx_arr,
                d_drift_tree_idx_arr,
                drift_tree_count);
    }

    return true;
}
