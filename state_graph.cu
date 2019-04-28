#include <bits/stdc++.h>

using namespace std;

class state_graph {
    public:

        bool is_stable;

        state_graph(int neighbor_limit) {
            is_stable = false;
            this->neighbor_limit = neighbor_limit;
        }

        int get_next_tree_id(int cur_tree_id) {
            if (map.find(cur_tree_id) == map.end()) {
                return -1;
            }

            node_t cur_node = map[cur_tree_id];
            int r = rand() % cur_node.total_count;
            int sum = 0;

            // weighted selection
            for (int i = 0; i < cur_node.counts.size(); i++) {
                sum += cur_node.counts[i];
                if (r < sum) {
                    return cur_node.neighbors[i];
                }
            }

            return -1;
        }

        void add_node(int key) {
            node_t cur_node;
            cur_node.key = key;
            cur_node.total_count = 0;

            map.insert({key, cur_node});
        }

        void add_edge(int from, int to) {
            if (map.find(from) == map.end()) {
                add_node(from);
            }
            if (map.find(to) == map.end()) {
                add_node(to);
            }

            node_t *cur_node = &map[from];
            vector<int>* nei = &(cur_node->neighbors);
            vector<int>* counts = &(cur_node->counts);

            auto iter = std::find(nei->begin(), nei->end(), to);
            if (iter == nei->end()) {
                nei->push_back(to);
                counts->push_back(1);
            } else {
                int pos = distance(nei->begin(), iter);
                (*counts)[pos] += 1;
            }

            cur_node->total_count++;
        }

        void remove_edge(int from, int to) {
            node_t *cur_node = &map[from];
            vector<int>* nei = &(cur_node->neighbors);
            vector<int>* counts = &(cur_node->counts);

            auto iter = std::find(nei->begin(), nei->end(), to);
            if (iter != nei->end()) {
                int pos = distance(nei->begin(), iter);
                cur_node->total_count -= (*counts)[pos];

                counts->erase(counts->begin() + pos);
                nei->erase(iter);
            }
        }

        vector<int>* get_neighbors(int key) {
            return &(map[key].neighbors);
        }

        vector<int>* get_counts(int key) {
            return &(map[key].counts);
        }

        int get_total_count(int key) {
            return map[key].total_count;
        }

    private:
        struct node_t {
            int key;
            vector<int> neighbors;
            vector<int> counts;
            int total_count;
        };

        int neighbor_limit;
        unordered_map<int, node_t> map;
};
