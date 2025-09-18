#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <random>
#include <fstream>
#include "louds-trie.hpp"

using namespace std;
using namespace std::chrono;
using namespace louds;

void verify_keys(Trie* trie, const vector<string>& expected_keys) {
    for (const string& key : expected_keys) {
        if (trie->lookup(key) == -1) {
            cerr << "ERROR: Expected key not found: " << key << endl;
            assert(false);
        }
    }
}

static void print_section(const string& title) {
    cout << "\n" << title << endl;
}

static void print_trie_summary(const char* name, const Trie& trie) {
    cout << name << ": "
         << trie.n_keys() << " keys, "
         << trie.n_nodes() << " nodes, "
         << trie.size() << " bytes" << endl;
}

static void print_approach(const char* title, double time_val, const Trie& merged, const char* unit) {
    cout << title << endl;
    cout << "  Time: " << time_val << " " << unit << endl;
    cout << "  Merged: " << merged.n_keys() << " keys, "
         << merged.n_nodes() << " nodes, "
         << merged.size() << " bytes" << endl << endl;
}

void test_basic_merge() {
    print_section("Basic Merge");
    
    Trie trie1;
    trie1.add("apple");
    trie1.add("banana");
    trie1.add("cherry");
    trie1.build();
    
    print_trie_summary("Trie 1", trie1);
    
    Trie trie2;
    trie2.add("apricot");
    trie2.add("blueberry");
    trie2.add("date");
    trie2.build();
    
    print_trie_summary("Trie 2", trie2);
    
    auto start = high_resolution_clock::now();
    Trie* merged1 = Trie::merge_trie(trie1, trie2);
    auto end = high_resolution_clock::now();
    double time1 = duration_cast<microseconds>(end - start).count();
    
    print_approach("Approach 1 (Extract-Merge-Rebuild):", time1, *merged1, "μs");
    
    start = high_resolution_clock::now();
    Trie* merged_linear = Trie::merge_trie_direct_linear(trie1, trie2);
    end = high_resolution_clock::now();
    double time_linear = duration_cast<microseconds>(end - start).count();
    
    print_approach("Approach 2 (Direct LOUDS Merge linear):", time_linear, *merged_linear, "μs");
    
    vector<string> expected = {"apple", "apricot", "banana", "blueberry", "cherry", "date"};

    verify_keys(merged1, expected);
    verify_keys(merged_linear, expected);    
    
    assert(merged1->n_keys() == merged_linear->n_keys());
    assert(merged1->n_nodes() == merged_linear->n_nodes());
    
    delete merged1;
    delete merged_linear;
    cout << "Basic merge test passed" << endl;
}

void test_overlapping_keys() {
    print_section("Overlapping Keys");
    
    Trie trie1;
    trie1.add("apple");
    trie1.add("banana");
    trie1.add("cherry");
    trie1.add("date");
    trie1.build();
    
    Trie trie2;
    trie2.add("banana");     // duplicate
    trie2.add("cherry");     // duplicate  
    trie2.add("elderberry");
    trie2.add("fig");
    trie2.build();
    
    print_trie_summary("Trie 1", trie1);
    print_trie_summary("Trie 2", trie2);
    
    auto start = high_resolution_clock::now();
    Trie* merged1 = Trie::merge_trie(trie1, trie2);
    auto end = high_resolution_clock::now();
    double t1 = duration_cast<microseconds>(end - start).count();
    
    start = high_resolution_clock::now();
    Trie* merged_linear = Trie::merge_trie_direct_linear(trie1, trie2);
    end = high_resolution_clock::now();
    double t2 = duration_cast<microseconds>(end - start).count();
    
    print_approach("Approach 1 (Extract-Merge-Rebuild):", t1, *merged1, "μs");
    print_approach("Approach 2 (Direct LOUDS Merge linear):", t2, *merged_linear, "μs");
     
    vector<string> expected = {"apple", "banana", "cherry", "date", "elderberry", "fig"};
    verify_keys(merged1, expected);
    verify_keys(merged_linear, expected);
    
    assert(merged1->n_keys() == 6);
    assert(merged_linear->n_keys() == 6);
    
    delete merged1;
    delete merged_linear;
    cout << "Overlapping keys test passed" << endl;
}

void test_empty_merge() {
    print_section("Empty Trie Merge");
    
    Trie trie1;
    trie1.add("alpha");
    trie1.add("beta");
    trie1.add("gamma");
    trie1.build();
    
    Trie trie2;
    trie2.build();
    
    print_trie_summary("Trie 1", trie1);
    print_trie_summary("Trie 2", trie2);
    
    auto start = high_resolution_clock::now();
    Trie* merged1 = Trie::merge_trie(trie1, trie2);
    auto end = high_resolution_clock::now();
    double t1 = duration_cast<microseconds>(end - start).count();
    
    start = high_resolution_clock::now();
    Trie* merged_linear = Trie::merge_trie_direct_linear(trie2, trie1);
    end = high_resolution_clock::now();
    double t2 = duration_cast<microseconds>(end - start).count();
    
    print_approach("Approach 1 (Extract-Merge-Rebuild):", t1, *merged1, "μs");
    print_approach("Approach 2 (Direct LOUDS Merge linear):", t2, *merged_linear, "μs");
     
    assert(merged1->n_keys() == 3);
    assert(merged_linear->n_keys() == 3);
    
    vector<string> expected = {"alpha", "beta", "gamma"};
    verify_keys(merged1, expected);
    verify_keys(merged_linear, expected);
    
    delete merged1;
    delete merged_linear;
    cout << "Empty merge test passed" << endl;
}

void test_common_prefixes() {
    print_section("Common Prefixes");
    
    Trie trie1;
    trie1.add("car");
    trie1.add("card");
    trie1.add("care");
    trie1.add("careful");
    trie1.add("carefully");
    trie1.build();
    
    Trie trie2;
    trie2.add("car");
    trie2.add("career");
    trie2.add("cargo");
    trie2.add("carrot");
    trie2.add("carry");
    trie2.build();
    
    print_trie_summary("Trie 1", trie1);
    print_trie_summary("Trie 2", trie2);
    
    auto start = high_resolution_clock::now();
    Trie* merged1 = Trie::merge_trie(trie1, trie2);
    auto end = high_resolution_clock::now();
    double t1 = duration_cast<microseconds>(end - start).count();
    
    start = high_resolution_clock::now();
    Trie* merged_linear = Trie::merge_trie_direct_linear(trie1, trie2);
    end = high_resolution_clock::now();
    double t2 = duration_cast<microseconds>(end - start).count();
    
    print_approach("Approach 1 (Extract-Merge-Rebuild):", t1, *merged1, "μs");
    print_approach("Approach 2 (Direct LOUDS Merge linear):", t2, *merged_linear, "μs");
 
     
    vector<string> expected = {
        "car", "card", "care", "career", "careful", 
        "carefully", "cargo", "carrot", "carry"
    };
     
    verify_keys(merged1, expected);
    verify_keys(merged_linear, expected);
     
    assert(merged1->n_keys() == 9);
    assert(merged_linear->n_keys() == 9);  
 
    delete merged1;
    delete merged_linear;
    cout << "Common prefixes test passed" << endl;
}

void test_performance_comparison() {
    print_section("Performance Comparison");
     
    vector<string> words1, words2;
     
    vector<string> prefixes = {"app", "ban", "car", "dat", "ele", "fig", "gra", "hon"};
    vector<string> suffixes = {"", "le", "ana", "rot", "eful", "efully", "ing", "ed", "er"};
     
    for (const auto& prefix : prefixes) {
        for (const auto& suffix : suffixes) {
            if ((prefix[0] - 'a') % 2 == 0) {
                words1.push_back(prefix + suffix);
            } else {
                words2.push_back(prefix + suffix);
            }
        }
    }
     
    for (int i = 0; i < 10; i++) {
        string word = "common" + to_string(i);
        words1.push_back(word);
        words2.push_back(word);
    }
     
    sort(words1.begin(), words1.end());
    sort(words2.begin(), words2.end());
     
    Trie trie1, trie2;
    for (const auto& word : words1) {
        trie1.add(word);
    }
    trie1.build();
     
    for (const auto& word : words2) {
        trie2.add(word);
    }
    trie2.build();
     
    print_trie_summary("Trie 1", trie1);
    print_trie_summary("Trie 2", trie2);
     
    auto start = high_resolution_clock::now();
    Trie* merged1 = Trie::merge_trie(trie1, trie2);
    auto end = high_resolution_clock::now();
    double time1 = duration_cast<microseconds>(end - start).count();
     
    start = high_resolution_clock::now();
    Trie* merged_linear = Trie::merge_trie_direct_linear(trie1, trie2);
    end = high_resolution_clock::now();
    double time_linear = duration_cast<microseconds>(end - start).count();
     
    print_approach("Approach 1 (Extract-Merge-Rebuild):", time1, *merged1, "μs");
    print_approach("Approach 2 (Direct LOUDS Merge linear):", time_linear, *merged_linear, "μs");
     
    cout << "Speedup: " << (time1 / time_linear) << "x" << endl << endl;
     
    assert(merged1->n_keys() == merged_linear->n_keys());
    assert(merged1->n_nodes() == merged_linear->n_nodes());
     
    delete merged1;
    delete merged_linear;
    cout << "Performance comparison complete" << endl;
}

static vector<string> read_sorted_unique_lines(const string& path) {
    ifstream in(path);
    vector<string> lines;
    if (!in) {
        cerr << "ERROR: Unable to open file: " << path << endl;
        assert(false);
    }
    string s;
    while (getline(in, s)) {
        lines.push_back(s);
    }
    sort(lines.begin(), lines.end());
    lines.erase(unique(lines.begin(), lines.end()), lines.end());
    return lines;
}

static void test_merge_from_files(const string& file1, const string& file2) {
    print_section("File-based Merge");

    vector<string> keys1 = read_sorted_unique_lines(file1);
    vector<string> keys2 = read_sorted_unique_lines(file2);

    Trie trie1, trie2;
    for (const auto& k : keys1) trie1.add(k);
    trie1.build();
    for (const auto& k : keys2) trie2.add(k);
    trie2.build();

    print_trie_summary("Trie 1", trie1);
    print_trie_summary("Trie 2", trie2);

    auto start = high_resolution_clock::now();
    Trie* merged1 = Trie::merge_trie(trie1, trie2);
    auto end = high_resolution_clock::now();
    double t1 = duration_cast<microseconds>(end - start).count();

    start = high_resolution_clock::now();
    Trie* merged2 = Trie::merge_trie_direct_linear(trie1, trie2);
    end = high_resolution_clock::now();
    double t2 = duration_cast<microseconds>(end - start).count();

    print_approach("Approach 1 (Extract-Merge-Rebuild):", t1, *merged1, "μs");
    print_approach("Approach 2 (Direct LOUDS Merge linear):", t2, *merged2, "μs");

    // vector<string> merged2_keys = merged2->get_all_keys();
    // vector<string> merged1_keys = merged1->get_all_keys();
    // for (const auto& k : merged2_keys) {
    //     std::cout << k << std::endl;
    // }
    // std::cout << "----------------" << std::endl;
    // for (const auto& k : merged1_keys) {
    //     std::cout << k << std::endl;
    // }

    vector<string> expected;
    expected.reserve(keys1.size() + keys2.size());
    set_union(keys1.begin(), keys1.end(), keys2.begin(), keys2.end(), back_inserter(expected));

    verify_keys(merged1, expected);
    verify_keys(merged2, expected);

    delete merged1;
    delete merged2;
    cout << "File-based merge test passed" << endl;
}

int main(int argc, char** argv) {
    cout << "Testing LOUDS Trie Merge" << endl;
    
    try {
        if (argc == 3) {
            test_merge_from_files(argv[1], argv[2]);
            cout << "All tests passed successfully" << endl;
            return 0;
        }
        test_basic_merge();
        test_overlapping_keys();
        test_empty_merge();
        test_common_prefixes();
        test_performance_comparison();        
        cout << "All tests passed successfully" << endl;
    } catch (const exception& e) {
        cerr << "Test failed with exception: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}