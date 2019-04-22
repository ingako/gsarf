#include <bits/stdc++.h>
#include <array>
#include <string>

using namespace std;

#define N 5

class LRU_state {
    public:
        LRU_state(int capacity, int distance_threshold) {
            this->capacity = capacity;
            this->distance_threshold = distance_threshold;
        }

        array<char, N> get(array<char, N> target_pattern) {

            string target_key(std::begin(target_pattern), std::end(target_pattern));

            if (map.find(target_key) != map.end()) {
                update_queue(target_key);

                print_queue();

                return target_pattern;
            }

            // find the smallest edit distance

            int min_edit_distance = INT_MAX;
            int max_freq = 0;
            array<char, N> closest_pattern = { '3' };

            for (auto cur_pair : queue) {
                array<char, N> cur_pattern = cur_pair.pattern;
                
                int cur_freq = cur_pair.freq;
                int cur_edit_distance = 0;

                bool update_flag = true;
                for (int i = 0; i < N; i++) {
                    if (cur_pattern[i] == target_pattern[i]) {
                        continue;
                    }

                    // tree with drift must be unset
                    if (cur_pattern[i] == '1' && target_pattern[i] == '2') {
                        update_flag = false;
                        break;
                    }

                    cur_edit_distance++;

                    if (cur_edit_distance > distance_threshold
                            || cur_edit_distance > min_edit_distance) {
                        update_flag = false;
                        break;
                    }
                }

                if (!update_flag) {
                    continue;
                }
                
                if (min_edit_distance == cur_edit_distance
                        && cur_freq > max_freq) {
                    continue;
                }

                min_edit_distance = cur_edit_distance;
                closest_pattern = cur_pattern;
            }


            if (min_edit_distance == INT_MAX) {
                // add new pattern to queue
                // return { 3 }
                put(target_pattern);

            } else {
                // update existing pattern in queue 
                // return closest_key

                string closest_key(std::begin(closest_pattern),
                        std::end(closest_pattern));
                update_queue(closest_key);
            }

            print_queue();

            return closest_pattern;
        }

        void print_queue() {
            list<state>::iterator it;
            for (it = queue.begin(); it != queue.end(); it++) {
                array<char, N> cur_pattern = it->pattern;
                int freq = it->freq;
                for (int i = 0; i < N; i++) {
                    cout << cur_pattern[i]; 
                }
                cout << ":" << freq << "->";
            }
            cout << endl;
        }

        void update_queue(string pattern_key) {
            auto pos = map[pattern_key];
            auto res = *pos;
            res.freq++;

            queue.erase(pos);
            queue.push_front(res);
            map[pattern_key] = queue.begin();
        }

        void put(array<char, N> pattern) {
            string key(std::begin(pattern), std::end(pattern));

            queue.emplace_front(pattern, 1, 1);
            map[key] = queue.begin();

            while (queue.size() > this->capacity) {
                array<char, N> rm_pattern = queue.back().pattern;
                string rm_pattern_str(std::begin(rm_pattern), std::end(rm_pattern));
                map.erase(rm_pattern_str);

                queue.pop_back();
            }
        }

    private:
        struct state {
            array<char, N> pattern;
            int val;
            int freq;
            state(array<char, N> pattern, int val, int freq) : pattern(pattern),
            val(val), freq(freq) {}
        };

        list<state> queue;
        unordered_map<string, list<state>::iterator> map;
        int capacity;
        int distance_threshold; 
};

int main() {
    LRU_state* state_queue = new LRU_state(4, 0);

    array<char, N> n0 = {'0', '0', '0', '0', '0'};
    array<char, N> n1 = {'0', '0', '0', '0', '1'};
    array<char, N> n2 = {'0', '0', '0', '1', '0'};
    array<char, N> n3 = {'0', '0', '0', '1', '1'};
    array<char, N> n4 = {'0', '0', '1', '0', '0'};

    cout << "get 0" << endl;
    state_queue->get(n0);

    cout << "\nget 1" << endl;
    state_queue->get(n1);

    cout << "\nget 2" << endl;
    state_queue->get(n2);

    cout << "\nget 0" << endl;
    state_queue->get(n0);

    cout << "\nget 3" << endl;
    state_queue->get(n3);

    cout << "\nget 4" << endl;
    state_queue->get(n4);

    return 0;
}
