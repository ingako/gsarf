#include "LRU_state.cu"

int main() {
    LRU_state* state_queue = new LRU_state(4, 0);

    vector<char> n0 = {'0', '0', '0', '0', '0'};
    vector<char> n1 = {'0', '0', '0', '0', '1'};
    vector<char> n2 = {'0', '0', '0', '1', '0'};
    vector<char> n3 = {'0', '0', '0', '1', '1'};
    vector<char> n4 = {'0', '0', '1', '0', '0'};
    
    vector<vector<char>> states = {n0, n1, n2, n0,n3, n4};
    
    vector<string> expected_result = {
        "00000:1->",
        "00001:1->00000:1->",
        "00010:1->00001:1->00000:1->",
        "00000:2->00010:1->00001:1->",
        "00011:1->00000:2->00010:1->00001:1->",
        "00100:1->00011:1->00000:2->00010:1->"
    };
    
    for (int i = 0; i < states.size(); i++) {
        vector<char> closest_state = state_queue->get_closest_state(states[i]);
        if (closest_state.size() == 0) {
            state_queue->enqueue(states[i]);
        } else {
            state_queue->update_queue(states[i]);
        }

        string cur_state_queue = state_queue->to_string();
        assert(cur_state_queue == expected_result[i]);
    }
    
    return 0;
}
