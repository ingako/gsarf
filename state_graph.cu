#include <bits/stdc++.h>

using namespace std;

class state_graph {
    public:
        state_graph() {}

        void add_node(string key) {
            node_t cur_node;
            cur_node.key = key;
            cur_node.total_count = 0;

            map.insert({key, cur_node});
        }

        bool add_edge(string from, string to) {
            if (map.find(from) == map.end()) {
                return false;
            }
            if (map.find(to) == map.end()) {
                return false;
            }

            map[from].neighbors.push_back(to);
            return true;
        }

        bool remove_edge(string from, string to) {
            vector<string> *neighbors = get_neighbors(from);
            auto iter = std::find(neighbors->begin(), neighbors->end(), to);

            if (iter == neighbors->end()) {
                return false;
            }

            neighbors->erase(iter);
            return true;
        }

        vector<string>* get_neighbors(string key) {
            return &map[key].neighbors;
        }

        vector<int>* get_counts(string key) {
            return &map[key].counts;
        }

        int get_total_count(string key) {
            return map[key].total_count;
        }

    private:
        struct node_t {
            string key;
            vector<string> neighbors;
            vector<int> counts;
            int total_count;
        };

        unordered_map<string, node_t> map;
};

int main() {
    state_graph* graph = new state_graph();
    graph->add_node("00000");
    graph->add_node("00001");
    graph->add_node("00010");

    vector<string> *nei = graph->get_neighbors("00000");
    assert(nei->size() == 0);

    graph->add_edge("00000", "00001");
    assert(nei->size() == 1);

    graph->add_edge("00000", "00010");
    assert(nei->size() == 2);

    graph->remove_edge("00000", "00001");
    assert(nei->size() == 1);

    graph->remove_edge("00000", "00010");
    assert(nei->size() == 0);

    return 0;
}
