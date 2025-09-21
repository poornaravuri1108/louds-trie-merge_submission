CXX       ?= g++
CXXFLAGS  ?= -O3 -std=c++17 -Wall -Wextra -march=native -mbmi2 -mpopcnt
SIMDFLAGS ?= -mavx2
LDFLAGS   ?=

HDRS = louds-trie.hpp
BASE = louds-trie-v1.cpp
SIMD = louds-trie.cpp
TEST = test.cpp

.PHONY: all clean compare
all: merge_v1 merge_simd

merge_v1: test_v1.o v1.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

test_v1.o: $(TEST) $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(TEST) -o $@

v1.o: $(BASE) $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(BASE) -o $@

merge_simd: test_simd.o simd.o
	$(CXX) $(CXXFLAGS) $(SIMDFLAGS) $^ -o $@ $(LDFLAGS)

test_simd.o: $(TEST) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SIMDFLAGS) -DUSE_SIMD -c $(TEST) -o $@

simd.o: $(SIMD) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SIMDFLAGS) -c $(SIMD) -o $@

compare: merge_v1 merge_simd
	@echo "Baseline (v1):"; ./merge_v1 | tee baseline.out >/dev/null
	@echo "SIMD:";         ./merge_simd | tee simd.out      >/dev/null
	@echo ""
	@echo "Summary:"
	@echo "- Direct LOUDS merge time in 'Performance Comparison' (μs):"
	@echo -n "  baseline: "; awk -F= '/^MERGE_LINEAR_TIME_US=/{print $$2}' baseline.out | tail -1
	@echo -n "  simd:     "; awk -F= '/^MERGE_LINEAR_TIME_US=/{print $$2}' simd.out     | tail -1
	@b=$$(awk -F= '/^MERGE_LINEAR_TIME_US=/{print $$2}' baseline.out | tail -1); \
	 s=$$(awk -F= '/^MERGE_LINEAR_TIME_US=/{print $$2}' simd.out     | tail -1); \
	 echo -n "  speedup:  "; awk -v b="$$b" -v s="$$s" 'BEGIN{if(b>0 && s>0) printf "%.2fx\n", b/s; else print "NA"}'
	@echo "- Lookup time (100 iters) (ms):"
	@echo -n "  baseline: "; awk -F= '/^LOOKUP_100X_MS=/{print $$2}' baseline.out | tail -1
	@echo -n "  simd:     "; awk -F= '/^LOOKUP_100X_MS=/{print $$2}' simd.out     | tail -1
	@b=$$(awk -F= '/^LOOKUP_100X_MS=/{print $$2}' baseline.out | tail -1); \
	 s=$$(awk -F= '/^LOOKUP_100X_MS=/{print $$2}' simd.out     | tail -1); \
	 echo -n "  speedup:  "; awk -v b="$$b" -v s="$$s" 'BEGIN{if(b>0 && s>0) printf "%.2fx\n", b/s; else print "NA"}'

clean:
	rm -f *.o merge_v1 merge_simd *.s *.out
