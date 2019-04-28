#include "state_graph.cu"

int main() {
    state_graph* graph = new state_graph(15);
    graph->add_node(0);
    graph->add_node(1);
    graph->add_node(2);

    vector<int> *nei = graph->get_neighbors(0);
    vector<int> *counts = graph->get_counts(0);
    assert(nei->size() == 0);

    graph->add_edge(0, 1);
    assert(nei->size() == 1);

    graph->add_edge(0, 2);
    assert(nei->size() == 2);

    graph->add_edge(0, 1);
    graph->add_edge(0, 1);
    assert((*counts)[0] == 3);
    assert(graph->get_total_count(0) == 4);

    int next_id = graph->get_next_tree_id(0);
    cout << "get_next_tree_idx(0): " << next_id << endl;

    graph->remove_edge(0, 1);
    assert(nei->size() == 1);
    assert(graph->get_total_count(0) == 1);

    graph->add_edge(0, 2);
    assert(nei->size() == 1);
    assert((*nei)[0] = 2);
    assert((*counts)[0] == 2);
    assert(graph->get_total_count(0) == 2);

    graph->remove_edge(0, 2);
    assert(nei->size() == 0);

    return 0;
}
