void detect_drifts(
        int* h_tree_active_status,
        vector<char>& target_state,
        int* h_tree_error_count,
        int* d_tree_error_count,
        int& warning_tree_count,
        int* h_warning_tree_idx_arr,
        int& drift_tree_count,
        int* h_drift_tree_idx_arr,
        vector<int>& warning_tree_id_list,
        vector<int>& drift_tree_id_list,
        int* forest_idx_to_tree_id,
        ADWIN** warning_detectors,
        ADWIN** drift_detectors) {

    gpuErrchk(cudaMemcpy((void *) h_tree_error_count, (void *) d_tree_error_count,
                TREE_COUNT * sizeof(int), cudaMemcpyDeviceToHost));

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
}
