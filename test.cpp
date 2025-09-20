#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <random>
#include <fstream>
#include "louds-trie.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;
using namespace std::chrono;
using namespace louds;

struct TestConfig {
    bool use_gpu = false;
    bool benchmark = false;
    bool verbose = false;
};

TestConfig g_config;

bool check_cuda_available() {
#ifdef USE_CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error == cudaSuccess && device_count > 0) {
        if (g_config.verbose) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            cout << "CUDA Device: " << prop.name << endl;
            cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
            cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n" << endl;
        }
        return true;
    }
#endif
    return false;
}

Trie* merge_tries(const Trie& t1, const Trie& t2, int approach) {
    if (approach == 1) {
        return Trie::merge_trie(t1, t2);
    } else if (approach == 2) {
        return Trie::merge_trie_direct_linear(t1, t2);
    } else if (approach == 3) {
#ifdef USE_CUDA
        if (g_config.use_gpu) {
            return Trie::merge_trie_direct_linear_cuda(t1, t2);
        } else {
            cout << "GPU requested but not enabled in this run" << endl;
            return Trie::merge_trie_direct_linear(t1, t2);
        }
#else
        cout << "GPU not available (compiled without CUDA)" << endl;
        return Trie::merge_trie_direct_linear(t1, t2);
#endif
    }
    return nullptr;
}

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
    cout << string(title.length(), '=') << endl;
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
    cout << "  Result: " << merged.n_keys() << " keys, "
         << merged.n_nodes() << " nodes, "
         << merged.size() << " bytes" << endl << endl;
}

void run_and_verify_approaches(const Trie& t1, const Trie& t2) {
    // Approach 1: Extract-Merge-Rebuild (Baseline)
    auto start = high_resolution_clock::now();
    Trie* merged1 = merge_tries(t1, t2, 1);
    auto end = high_resolution_clock::now();
    double time1 = duration_cast<microseconds>(end - start).count();
    print_approach("Approach 1 (Extract-Merge-Rebuild):", time1, *merged1, "μs");

    // Approach 2: Direct LOUDS Merge (CPU)
    start = high_resolution_clock::now();
    Trie* merged2 = merge_tries(t1, t2, 2);
    end = high_resolution_clock::now();
    double time2 = duration_cast<microseconds>(end - start).count();
    print_approach("Approach 2 (Direct LOUDS CPU):", time2, *merged2, "μs");

    // Verify correctness between CPU approaches
    assert(merged1->n_keys() == merged2->n_keys());
    assert(merged1->n_nodes() == merged2->n_nodes());

    // Approach 3: Direct LOUDS Merge (GPU)
    if (g_config.use_gpu) {
        start = high_resolution_clock::now();
        Trie* merged3 = merge_tries(t1, t2, 3);
        end = high_resolution_clock::now();
        double time3 = duration_cast<microseconds>(end - start).count();
        print_approach("Approach 3 (Direct LOUDS GPU):", time3, *merged3, "μs");

        cout << "Speedup GPU vs CPU (Direct): " << (time2 / time3) << "x" << endl;

        // Verify GPU result against CPU result
        assert(merged3->n_keys() == merged2->n_keys());
        assert(merged3->n_nodes() == merged2->n_nodes());
        delete merged3;
    }

    delete merged1;
    delete merged2;
}

void test_basic_merge() {
    print_section("Basic Merge Test");
    
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
    
    run_and_verify_approaches(trie1, trie2);
    
    vector<string> expected = {"apple", "apricot", "banana", "blueberry", "cherry", "date"};
    verify_keys(merge_tries(trie1, trie2, 3), expected);
    cout << "Basic merge test passed" << endl;
}

void test_performance_comparison() {
    print_section("Performance Comparison");
    
    vector<int> sizes = {100, 1000, 10000};
    if (g_config.benchmark) {
        sizes.push_back(100000);
    }
    
    for (int size : sizes) {
        cout << "\nDataset size: " << size << " keys" << endl;
        
        Trie trie1, trie2;
        vector<string> words1, words2;
        
        for (int i = 0; i < size; i++) {
            string word = "word" + to_string(i * 2);
            words1.push_back(word);
        }
        sort(words1.begin(), words1.end());
        for (const auto& word : words1) {
            trie1.add(word);
        }
        trie1.build();
        
        for (int i = 0; i < size; i++) {
            string word = "word" + to_string(i * 2 + 1);
            words2.push_back(word);
        }
        for (int i = 0; i < size/10; i++) {
            string word = "word" + to_string(i * 2);
            words2.push_back(word);
        }
        sort(words2.begin(), words2.end());
        words2.erase(unique(words2.begin(), words2.end()), words2.end());
        for (const auto& word : words2) {
            trie2.add(word);
        }
        trie2.build();
        
        print_trie_summary("  Trie 1", trie1);
        print_trie_summary("  Trie 2", trie2);
        
        run_and_verify_approaches(trie1, trie2);
    }
}

static vector<string> read_sorted_unique_lines(const string& path) {
    ifstream in(path);
    vector<string> lines;
    if (!in) {
        cerr << "ERROR: Unable to open file: " << path << endl;
        return lines;
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
    print_section("File-based Merge Test");

    vector<string> keys1 = read_sorted_unique_lines(file1);
    vector<string> keys2 = read_sorted_unique_lines(file2);

    if (keys1.empty() || keys2.empty()) {
        cout << "Skipping file test - files not found" << endl;
        return;
    }

    Trie trie1, trie2;
    for (const auto& k : keys1) trie1.add(k);
    trie1.build();
    for (const auto& k : keys2) trie2.add(k);
    trie2.build();

    print_trie_summary("Trie 1", trie1);
    print_trie_summary("Trie 2", trie2);

    run_and_verify_approaches(trie1, trie2);
    cout << "File-based merge test passed" << endl;
}

void print_usage() {
    cout << "Usage: ./merge-test [options] [file1 file2]" << endl;
    cout << "Options:" << endl;
    cout << "  --gpu         Enable GPU acceleration (if compiled with CUDA)" << endl;
    cout << "  --benchmark   Run extended benchmarks" << endl;
    cout << "  --verbose     Enable verbose output" << endl;
    cout << "  --help        Show this help message" << endl;
}

int main(int argc, char** argv) {
    cout << "LOUDS Trie Merge Testing Suite" << endl;
    
    string file1, file2;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--gpu") {
            g_config.use_gpu = true;
        } else if (arg == "--benchmark") {
            g_config.benchmark = true;
        } else if (arg == "--verbose") {
            g_config.verbose = true;
        } else if (arg == "--help") {
            print_usage();
            return 0;
        } else if (file1.empty()) {
            file1 = arg;
        } else if (file2.empty()) {
            file2 = arg;
        }
    }
    
    if (g_config.use_gpu) {
        if (check_cuda_available()) {
            cout << "CUDA is available and will be used" << endl;
        } else {
            cout << "CUDA not available - falling back to CPU" << endl;
            g_config.use_gpu = false;
        }
    } else {
        cout << "Running CPU-only tests (use --gpu for GPU acceleration)" << endl;
    }
    
    try {
        if (!file1.empty() && !file2.empty()) {
            test_merge_from_files(file1, file2);
        } else {
            test_basic_merge();
            test_performance_comparison();
        }
        
        cout << "All tests passed successfully!" << endl;
    } catch (const exception& e) {
        cerr << "Test failed with exception: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
