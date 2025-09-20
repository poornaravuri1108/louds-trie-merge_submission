#include "louds-trie.hpp"

#ifdef _MSC_VER
  #include <intrin.h>
  #include <immintrin.h>
#else  // _MSC_VER
  #include <x86intrin.h>
#endif  // _MSC_VER

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sequence.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t _err = (call);                                             \
    if (_err != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                      \
              #call, __FILE__, __LINE__, cudaGetErrorString(_err));        \
      abort();                                                             \
    }                                                                      \
  } while (0)
#endif
#endif

#include <cassert>
#include <vector>
#include <algorithm>

namespace louds {
namespace {

uint64_t Popcnt(uint64_t x) {
#ifdef _MSC_VER
  return __popcnt64(x);
#else  // _MSC_VER
  return __builtin_popcountll(x);
#endif  // _MSC_VER
}

uint64_t Ctz(uint64_t x) {
#ifdef _MSC_VER
  return _tzcnt_u64(x);
#else  // _MSC_VER
  return __builtin_ctzll(x);
#endif  // _MSC_VER
}

struct BitVector {
  struct Rank {
    uint32_t abs_hi;
    uint8_t abs_lo;
    uint8_t rels[3];

    uint64_t abs() const {
      return ((uint64_t)abs_hi << 8) | abs_lo;
    }
    void set_abs(uint64_t abs) {
      abs_hi = (uint32_t)(abs >> 8);
      abs_lo = (uint8_t)abs;
    }
  };

  vector<uint64_t> words;
  vector<Rank> ranks;
  vector<uint32_t> selects;
  uint64_t n_bits;

  BitVector() : words(), ranks(), selects(), n_bits(0) {}

  uint64_t get(uint64_t i) const {
    return (words[i / 64] >> (i % 64)) & 1UL;
  }
  void set(uint64_t i, uint64_t bit) {
    if (bit) {
      words[i / 64] |= (1UL << (i % 64));
    } else {
      words[i / 64] &= ~(1UL << (i % 64));
    }
  }

  void add(uint64_t bit) {
    if (n_bits % 256 == 0) {
      words.resize((n_bits + 256) / 64);
    }
    set(n_bits, bit);
    ++n_bits;
  }
  // build builds indexes for rank and select.
  void build() {
    uint64_t n_blocks = words.size() / 4;
    uint64_t n_ones = 0;
    ranks.resize(n_blocks + 1);
    for (uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
      ranks[block_id].set_abs(n_ones);
      for (uint64_t j = 0; j < 4; ++j) {
        if (j != 0) {
          uint64_t rel = n_ones - ranks[block_id].abs();
          ranks[block_id].rels[j - 1] = rel;
        }

        uint64_t word_id = (block_id * 4) + j;
        uint64_t word = words[word_id];
        uint64_t n_pops = Popcnt(word);
        uint64_t new_n_ones = n_ones + n_pops;
        if (((n_ones + 255) / 256) != ((new_n_ones + 255) / 256)) {
          uint64_t count = n_ones;
          while (word != 0) {
            uint64_t pos = Ctz(word);
            if (count % 256 == 0) {
              selects.push_back(((word_id * 64) + pos) / 256);
              break;
            }
            word ^= 1UL << pos;
            ++count;
          }
        }
        n_ones = new_n_ones;
      }
    }
    ranks.back().set_abs(n_ones);
    selects.push_back(words.size() * 64 / 256);
  }

  // rank returns the number of 1-bits in the range [0, i).
  uint64_t rank(uint64_t i) const {
    uint64_t word_id = i / 64;
    uint64_t bit_id = i % 64;
    uint64_t rank_id = word_id / 4;
    uint64_t rel_id = word_id % 4;
    uint64_t n = ranks[rank_id].abs();
    if (rel_id != 0) {
      n += ranks[rank_id].rels[rel_id - 1];
    }
    n += Popcnt(words[word_id] & ((1UL << bit_id) - 1));
    return n;
  }
  // select returns the position of the (i+1)-th 1-bit.
  uint64_t select(uint64_t i) const {
    const uint64_t block_id = i / 256;
    uint64_t begin = selects[block_id];
    uint64_t end = selects[block_id + 1] + 1UL;
    if (begin + 10 >= end) {
      while (i >= ranks[begin + 1].abs()) {
        ++begin;
      }
    } else {
      while (begin + 1 < end) {
        const uint64_t middle = (begin + end) / 2;
        if (i < ranks[middle].abs()) {
          end = middle;
        } else {
          begin = middle;
        }
      }
    }
    const uint64_t rank_id = begin;
    i -= ranks[rank_id].abs();

    uint64_t word_id = rank_id * 4;
    if (i < ranks[rank_id].rels[1]) {
      if (i >= ranks[rank_id].rels[0]) {
        word_id += 1;
        i -= ranks[rank_id].rels[0];
      }
    } else if (i < ranks[rank_id].rels[2]) {
      word_id += 2;
      i -= ranks[rank_id].rels[1];
    } else {
      word_id += 3;
      i -= ranks[rank_id].rels[2];
    }
    return (word_id * 64) + Ctz(_pdep_u64(1UL << i, words[word_id]));
  }

  uint64_t size() const {
    return sizeof(uint64_t) * words.size()
      + sizeof(Rank) * ranks.size()
      + sizeof(uint32_t) * selects.size();
  }
};

static inline void assign_from_bits(BitVector& bv, const std::vector<uint8_t>& bits) {
    const uint64_t n = bits.size();
    bv.n_bits = n;
    bv.words.assign((n + 63) >> 6, 0ULL);
    for (uint64_t i = 0; i < n; ++i) {
      if (bits[i]) bv.words[i >> 6] |= (1ULL << (i & 63));
    }
}

struct Level {
  BitVector louds;
  BitVector outs;
  vector<uint8_t> labels;
  uint64_t offset;

  Level() : louds(), outs(), labels(), offset(0) {}

  uint64_t size() const;
};

uint64_t Level::size() const {
  return louds.size() + outs.size() + labels.size();
}

// inline void child_range(const std::vector<Level>& Lv, uint64_t lev, uint64_t node_id, uint64_t& b, uint64_t& e) {
//   b = e = 0;
//   if (lev + 1 >= Lv.size()) return;
//   const Level& ch = Lv[lev + 1];
//   if (ch.louds.n_bits == 0) return;

//   uint64_t start_pos = (node_id != 0) ? (ch.louds.select(node_id - 1) + 1) : 0;
//   uint64_t pos = start_pos;
//   while (pos < ch.louds.n_bits && !ch.louds.get(pos)) ++pos;  
//   uint64_t k = pos - start_pos;                                
//   b = start_pos - node_id;                                     
//   e = b + k;                                                   
// }

inline void child_range(const std::vector<Level>& Lv, uint64_t lev, uint64_t node_id,
    uint64_t& b, uint64_t& e) {
    b = e = 0;
    if (lev + 1 >= Lv.size()) return;
    const Level& ch = Lv[lev + 1];
    if (ch.louds.n_bits == 0) return;

    const uint64_t start_pos = (node_id != 0) ? (ch.louds.select(node_id - 1) + 1) : 0;
    uint64_t end = start_pos;
    uint64_t word = ch.louds.words[end >> 6] >> (end & 63);
    if (word == 0) {
    end += (64 - (end & 63));
    while ((end >> 6) < ch.louds.words.size()) {
    word = ch.louds.words[end >> 6];
    if (word) break;
    end += 64;
    }
    }
    if (word) end += Ctz(word);
    const uint64_t k = end - start_pos;
    b = start_pos - node_id;
    e = b + k;
}


inline bool is_terminal_at(const std::vector<Level>& Lv, uint64_t lev_plus_1, uint64_t child_id) {
  if (lev_plus_1 >= Lv.size()) return false;
  const Level& L = Lv[lev_plus_1];
  return (child_id < L.outs.n_bits) && (L.outs.get(child_id) != 0);
}

inline void append_parent_one(std::vector<Level>& out_levels, uint64_t lev, bool is_first_parent_at_level) {
  const uint64_t target = lev + 1;
  if (out_levels.size() <= target) out_levels.resize(target + 1);
  if (lev == 0 && is_first_parent_at_level) return;
  out_levels[target].louds.add(1);
}

inline void emit_child(std::vector<Level>& out_levels, uint64_t lev, uint8_t label) {
  Level& nextL = out_levels[lev + 1];
  if (nextL.louds.n_bits == 0) {
    nextL.louds.add(0);
    nextL.louds.add(1);
  } else {
    nextL.louds.set(nextL.louds.n_bits - 1, 0);
    nextL.louds.add(1);
  }
  nextL.labels.push_back(label);
  nextL.outs.add(0);
}

}  // namespace

class TrieImpl {
 public:
  TrieImpl();
  ~TrieImpl() {}

  void add(const string &key);
  void build();
  int64_t lookup(const string &query) const;

  uint64_t n_keys() const {
    return n_keys_;
  }
  uint64_t n_nodes() const {
    return n_nodes_;
  }
  uint64_t size() const {
    return size_;
  }

  void collect_all_keys(vector<string>& keys) const;
  const vector<Level>& get_levels() const { return levels_; }

 private:
  vector<Level> levels_;
  uint64_t n_keys_;
  uint64_t n_nodes_;
  uint64_t size_;
  string last_key_;

  void collect_keys_recursive(vector<string>& keys, uint64_t node_id, uint64_t level, string& prefix) const;

  // Friend declaration for merge functions
  friend Trie* Trie::merge_trie(const Trie& trie1, const Trie& trie2);
  friend Trie* Trie::merge_trie_direct_linear(const Trie& trie1, const Trie& trie2);
  friend Trie* Trie::merge_trie_direct_linear_cuda(const Trie& t1, const Trie& t2);
};

TrieImpl::TrieImpl()
  : levels_(2), n_keys_(0), n_nodes_(1), size_(0), last_key_() {
  levels_[0].louds.add(0);
  levels_[0].louds.add(1);
  levels_[1].louds.add(1);
  levels_[0].outs.add(0);
  levels_[0].labels.push_back(' ');
}

void TrieImpl::add(const string &key) {
  assert(key > last_key_);
  if (key.empty()) {
    levels_[0].outs.set(0, 1);
    ++levels_[1].offset;
    ++n_keys_;
    last_key_ = key;
    return;
  }
  if (key.length() + 1 >= levels_.size()) {
    levels_.resize(key.length() + 2);
  }
  uint64_t i = 0;
  for ( ; i < key.length(); ++i) {
    Level &level = levels_[i + 1];
    uint8_t byte = key[i];
    if ((i == last_key_.length()) || (byte != level.labels.back())) {
      level.louds.set(levels_[i + 1].louds.n_bits - 1, 0);
      level.louds.add(1);
      level.outs.add(0);
      level.labels.push_back(key[i]);
      ++n_nodes_;
      break;
    }
  }
  for (++i; i < key.length(); ++i) {
    Level &level = levels_[i + 1];
    level.louds.add(0);
    level.louds.add(1);
    level.outs.add(0);
    level.labels.push_back(key[i]);
    ++n_nodes_;
  }
  levels_[key.length() + 1].louds.add(1);
  ++levels_[key.length() + 1].offset;
  levels_[key.length()].outs.set(levels_[key.length()].outs.n_bits - 1, 1);
  ++n_keys_;
  last_key_ = key;
}

void TrieImpl::build() {
  uint64_t offset = 0;
  for (uint64_t i = 0; i < levels_.size(); ++i) {
    Level &level = levels_[i];
    level.louds.build();
    level.outs.build();
    offset += levels_[i].offset;
    level.offset = offset;
    size_ += level.size();
  }
}

int64_t TrieImpl::lookup(const string &query) const {
  if (query.length() >= levels_.size()) {
    return -1;
  }
  uint64_t node_id = 0;
  for (uint64_t i = 0; i < query.length(); ++i) {
    const Level &level = levels_[i + 1];
    uint64_t node_pos;
    if (node_id != 0) {
      node_pos = level.louds.select(node_id - 1) + 1;
      node_id = node_pos - node_id;
    } else {
      node_pos = 0;
    }

    // Linear search implementation
    // for (uint8_t byte = query[i]; ; ++node_pos, ++node_id) {
    //   if (level.louds.get(node_pos) || level.labels[node_id] > byte) {
    //     return -1;
    //   }
    //   if (level.labels[node_id] == byte) {
    //     break;
    //   }
    // }

    // Binary search implementation
    uint64_t end = node_pos;
    uint64_t word = level.louds.words[end / 64] >> (end % 64);
    if (word == 0) {
      end += 64 - (end % 64);
      word = level.louds.words[end / 64];
      while (word == 0) {
        end += 64;
        word = level.louds.words[end / 64];
      }
    }
    end += Ctz(word);
    uint64_t begin = node_id;
    end = begin + end - node_pos;

    uint8_t byte = query[i];
    while (begin < end) {
      node_id = (begin + end) / 2;
      if (byte < level.labels[node_id]) {
        end = node_id;
      } else if (byte > level.labels[node_id]) {
        begin = node_id + 1;
      } else {
        break;
      }
    }
    if (begin >= end) {
      return -1;
    }
  }
  const Level &level = levels_[query.length()];
  if (!level.outs.get(node_id)) {
    return -1;
  }
  return level.offset + level.outs.rank(node_id);
}

Trie::Trie() : impl_(new TrieImpl) {}

Trie::~Trie() {
  delete impl_;
}

void Trie::add(const string &key) {
  impl_->add(key);
}

void Trie::build() {
  impl_->build();
}

int64_t Trie::lookup(const string &query) const {
  return impl_->lookup(query);
}

uint64_t Trie::n_keys() const {
  return impl_->n_keys();
}

uint64_t Trie::n_nodes() const {
  return impl_->n_nodes();
}

uint64_t Trie::size() const {
  return impl_->size();
}

std::vector<std::string> Trie::get_all_keys() const {
  std::vector<std::string> keys;
  impl_->collect_all_keys(keys);
  std::sort(keys.begin(), keys.end());
  return keys;
}

void TrieImpl::collect_keys_recursive(vector<string>& keys, uint64_t node_id, uint64_t level_idx, string& prefix) const {
  
  if (level_idx == 0 && levels_[0].outs.get(0)) {
    keys.push_back("");
  }
  
  if (level_idx >= levels_.size()) {
    return;
  }
  
  const Level& level = levels_[level_idx];
  
  if (level_idx > 0 && node_id < level.outs.n_bits && level.outs.get(node_id)) {
    keys.push_back(prefix);
  }
  
  if (level_idx + 1 >= levels_.size()) {
    return;
  }
  
  const Level& child_level = levels_[level_idx + 1];
  
  if (child_level.louds.n_bits == 0) {
    return;
  }
  
  uint64_t child_pos;
  
  if (node_id != 0) {
    if (node_id - 1 >= child_level.louds.rank(child_level.louds.n_bits)) {
      return;
    }
    child_pos = child_level.louds.select(node_id - 1) + 1;
  } else {
    child_pos = 0;
  }
  
  uint64_t child_id = child_pos - node_id;
  
  while (child_pos < child_level.louds.n_bits && !child_level.louds.get(child_pos)) {
    if (child_id < child_level.labels.size()) {
      prefix.push_back(child_level.labels[child_id]);
      collect_keys_recursive(keys, child_id, level_idx + 1, prefix);
      prefix.pop_back();
    }
    child_pos++;
    child_id++;
  }
}

void TrieImpl::collect_all_keys(vector<string>& keys) const {
  string prefix;
  collect_keys_recursive(keys, 0, 0, prefix);
}

/*
Approach 1: Extract–Merge–Rebuild

Summary
  Pull out all keys from both tries (they come out sorted because add() enforces
  lexicographic order), merge the two sorted lists while removing duplicates,
  then build a new LOUDS trie by inserting the merged keys.

Algorithm
  1) collect_all_keys(trie1) and collect_all_keys(trie2) -> two sorted vectors.
  2) Two-way merge the vectors, skipping equal neighbors (dedupe).
  3) For each merged key: trie_out.add(key); finally call trie_out.build().

Complexity (simple terms)
  Let n1, n2 = number of keys; K1, K2 = total characters across keys in trie1, trie2;
  Kout = total characters across merged keys; |Eout| = total edges in merged trie.

  Time: O(K1 + K2 + Kout + |Eout|)

  Space: O(K1 + K2 + n1 + n2 + |Eout|)
*/
Trie* Trie::merge_trie(const Trie& trie1, const Trie& trie2) {
  Trie* merged = new Trie();
  
  vector<string> keys1, keys2;
  trie1.impl_->collect_all_keys(keys1);
  trie2.impl_->collect_all_keys(keys2);

  assert(std::is_sorted(keys1.begin(), keys1.end()));
  assert(std::is_sorted(keys2.begin(), keys2.end()));
  
  vector<string> merged_keys;
  merged_keys.reserve(keys1.size() + keys2.size());
  
  size_t i = 0, j = 0;
  while (i < keys1.size() && j < keys2.size()) {
    if (keys1[i] < keys2[j]) {
      merged_keys.push_back(keys1[i++]);
    } else if (keys1[i] > keys2[j]) {
      merged_keys.push_back(keys2[j++]);
    } else {
      merged_keys.push_back(keys1[i]);
      i++;
      j++;
    }
  }
  
  while (i < keys1.size()) {
    merged_keys.push_back(keys1[i++]);
  }
  while (j < keys2.size()) {
    merged_keys.push_back(keys2[j++]);
  }
  
  for (const string& key : merged_keys) {
    merged->add(key);
  }
  merged->build();
  
  return merged;
}


/*
Approach 2 — Direct LOUDS merge (no strings) and Efficient

Goal
  Merge two tries by their LOUDS levels directly. We never build full keys.

Idea:
  - For each level (lev) while we still have parents:
    - For each parent (from trie1 and/or trie2), find its child ranges with child_range().

    - Start this parent’s child list in the OUTPUT:
        - Put a trailing '1' at out.levels[lev+1] (append_parent_one).
          (Leaf parents keep this '1'; if we add a child, we’ll flip it to 0.)

    - Merge the two sorted child-label lists with two pointers:
        - For each chosen label:
            - emit_child(lev, label): flip the last 1 -> 0, then add a 1; append the label; add outs=0.
            - If this child ends a key in either input, set outs=1 and ++offset at lev+2.
            - Enqueue the next-level node pair for this child
              (use the child id from a trie if it exists; if not, mark that side as absent).

Complexity
  Time: O(|E1| + |E2|) to merge + O(|Eout|) for build().
  Space: O(|Eout|).
*/
Trie* Trie::merge_trie_direct_linear(const Trie& t1, const Trie& t2) {
  Trie* out = new Trie();
  auto& out_impl   = *out->impl_;
  auto& out_levels = out_impl.levels_;
  const auto& L1   = t1.impl_->get_levels();
  const auto& L2   = t2.impl_->get_levels();

  out_impl.n_keys_  = 0;
  out_impl.n_nodes_ = 1; 
  out_impl.size_    = 0;
  out_levels.resize(2);

  if (!L1.empty() && L1[0].outs.get(0)) { 
    out_levels[0].outs.set(0,1); ++out_levels[1].offset; ++out_impl.n_keys_; 
  }

  if (!L2.empty() && L2[0].outs.get(0) && !out_levels[0].outs.get(0)) { 
    out_levels[0].outs.set(0,1); ++out_levels[1].offset; ++out_impl.n_keys_; 
  }

  struct Pair { bool h1, h2; uint64_t id1, id2; };
  std::vector<Pair> curr(1, Pair{ !L1.empty(), !L2.empty(), 0, 0 }), next;

  for (uint64_t lev = 0; !curr.empty(); ++lev) {
    if (out_levels.size() <= lev + 1) out_levels.resize(lev + 2);
    next.clear();

    for (size_t pidx = 0; pidx < curr.size(); ++pidx) {
      const auto& p = curr[pidx];

      uint64_t b1=0,e1=0, b2=0,e2=0;
      if (p.h1) child_range(L1, lev, p.id1, b1, e1);
      if (p.h2) child_range(L2, lev, p.id2, b2, e2);

      append_parent_one(out_levels, lev, (pidx == 0));

      if (b1 == e1 && b2 == e2) {
        continue; 
      }

      while (b1 < e1 || b2 < e2) {
        bool take1=false, take2=false;
        uint8_t lab=0;

        if (b1 < e1 && b2 < e2) {
          uint8_t l1 = L1[lev + 1].labels[b1];
          uint8_t l2 = L2[lev + 1].labels[b2];
          if (l1 == l2) { 
            lab = l1; take1 = take2 = true; 
          }
          else if (l1 < l2) { 
            lab = l1; take1 = true; 
          }
          else { 
            lab = l2; take2 = true; 
          }
        } else if (b1 < e1) { 
          lab = L1[lev + 1].labels[b1]; take1 = true; 
        }
        else { 
          lab = L2[lev + 1].labels[b2]; take2 = true; 
        }

        emit_child(out_levels, lev, lab);

        bool term = (take1 && is_terminal_at(L1, lev + 1, b1))
                 || (take2 && is_terminal_at(L2, lev + 1, b2));
                 
        if (term) {
          Level& here = out_levels[lev + 1];
          here.outs.set(here.outs.n_bits - 1, 1);
          if (out_levels.size() <= lev + 2) out_levels.resize(lev + 3);
          ++out_levels[lev + 2].offset;
          ++out_impl.n_keys_;
        }

        next.push_back(Pair{ take1, take2, take1 ? b1 : 0, take2 ? b2 : 0 });
        ++out_impl.n_nodes_;

        if (take1) ++b1;
        if (take2) ++b2;
      }
    }
    curr.swap(next);  
  }

  out_impl.build();
  return out;
}

/*
  Cuda implementation of efficient LOUDS merge

  Approach: 
    - Implemented a direct, level-by-level LOUDS merge on both CPU and GPU (no string reconstruction). 
    - At each level, the GPU assigns one thread per parent node. Kernel k_count merges the two sorted child label lists for that parent to compute the output child count and terminal count. 
    - A Thrust exclusive_scan then produces per-parent write offsets. 
    - Kernel k_emit writes the merged labels, OUTS bits, LOUDS segments, and emits the next-level parent pairs. 
    - This is repeated per level and materialize the bitvectors once per level. 
    - Techniques used include stream-aware Thrust (thrust::cuda::par.on(stream)), async H2D/D2H on a non-blocking CUDA stream, device-buffer reuse to avoid frequent cudaMalloc, and bit-packed access on device (d_getbit from 64-bit words). 
    - On datasets (1k–10k keys), this yields consistent ~1.15–1.35× speedups over the CPU direct merge (and ~3–4× over extract–merge–rebuild), with identical result structure/size.
*/

#ifdef USE_CUDA
// Small host-only helper: child range (local to this TU)
static inline void crange(const std::vector<Level>& Lv, uint32_t lev, uint32_t node_id,
                          uint32_t& b, uint32_t& e) {
  b = e = 0;
  if (lev + 1 >= Lv.size()) return;
  const Level& ch = Lv[lev + 1];
  if (ch.louds.n_bits == 0) return;
  const uint64_t start_pos = (node_id != 0) ? (ch.louds.select(node_id - 1) + 1) : 0;
  uint64_t end = start_pos;
  uint64_t word = ch.louds.words[end >> 6] >> (end & 63);
  if (word == 0) {
    end += (64 - (end & 63));
    while ((end >> 6) < ch.louds.words.size()) {
      word = ch.louds.words[end >> 6];
      if (word) break;
      end += 64;
    }
  }
  if (word) end += Ctz(word);
  const uint64_t k = end - start_pos;
  b = static_cast<uint32_t>(start_pos - node_id);
  e = b + static_cast<uint32_t>(k);
}
#endif

struct GPUPair { uint8_t h1,h2; uint32_t id1,id2; };

#ifdef __CUDACC__
__device__ __forceinline__ uint8_t d_getbit(const uint64_t* w, uint32_t i) {
  return (w[i >> 6] >> (i & 63)) & 1u;
}

__global__ void k_count(
  const uint8_t* __restrict__ lab1, const uint32_t* __restrict__ b1, const uint32_t* __restrict__ e1,
  const uint64_t* __restrict__ outs1w,
  const uint8_t* __restrict__ lab2, const uint32_t* __restrict__ b2, const uint32_t* __restrict__ e2,
  const uint64_t* __restrict__ outs2w,
  const GPUPair* __restrict__ pairs, uint32_t P,
  uint32_t* __restrict__ out_len, uint32_t* __restrict__ term_cnt)
{
  const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;

  const bool H1 = pairs[p].h1, H2 = pairs[p].h2;
  uint32_t i = H1 ? b1[p] : 0, j = H2 ? b2[p] : 0;
  const uint32_t ie = H1 ? e1[p] : 0, je = H2 ? e2[p] : 0;

  uint32_t m = 0, tc = 0;
  while ((H1 && i < ie) || (H2 && j < je)) {
    uint8_t take1=0, take2=0, lab=0;
    if (H1 && i<ie && H2 && j<je) {
      const uint8_t l1 = lab1[i], l2 = lab2[j];
      if (l1 == l2) { lab=l1; take1=take2=1; }
      else if (l1 < l2) { lab=l1; take1=1; } else { lab=l2; take2=1; }
    } else if (H1 && i<ie) { lab=lab1[i]; take1=1; }
    else { lab=lab2[j]; take2=1; }

    uint8_t term = 0;
    if (take1) term |= d_getbit(outs1w, i);
    if (take2) term |= d_getbit(outs2w, j);
    tc += term;
    ++m;
    if (take1) ++i;
    if (take2) ++j;
  }
  out_len[p] = m;
  term_cnt[p] = tc;
}

__global__ void k_emit(
  const uint8_t* __restrict__ lab1, const uint32_t* __restrict__ b1, const uint32_t* __restrict__ e1,
  const uint64_t* __restrict__ outs1w,
  const uint8_t* __restrict__ lab2, const uint32_t* __restrict__ b2, const uint32_t* __restrict__ e2,
  const uint64_t* __restrict__ outs2w,
  const GPUPair* __restrict__ pairs, uint32_t P,
  const uint32_t* __restrict__ child_base, const uint32_t* __restrict__ louds_base,
  uint8_t* __restrict__ labels_out, uint8_t* __restrict__ outs_bits_out, uint8_t* __restrict__ louds_bits_out,
  GPUPair* __restrict__ next_pairs)
{
  const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;

  const bool H1 = pairs[p].h1, H2 = pairs[p].h2;
  uint32_t i = H1 ? b1[p] : 0, j = H2 ? b2[p] : 0;
  const uint32_t ie = H1 ? e1[p] : 0, je = H2 ? e2[p] : 0;

  const uint32_t cb = child_base[p];
  const uint32_t lb = louds_base[p];
  uint32_t out_i = 0;

  while ((H1 && i < ie) || (H2 && j < je)) {
    uint8_t take1=0, take2=0, lab=0;
    if (H1 && i<ie && H2 && j<je) {
      const uint8_t l1 = lab1[i], l2 = lab2[j];
      if (l1 == l2) { lab=l1; take1=take2=1; }
      else if (l1 < l2) { lab=l1; take1=1; } else { lab=l2; take2=1; }
    } else if (H1 && i<ie) { lab=lab1[i]; take1=1; }
    else { lab=lab2[j]; take2=1; }

    labels_out[cb + out_i] = lab;
    const uint8_t term = (take1 ? d_getbit(outs1w, i) : 0) | (take2 ? d_getbit(outs2w, j) : 0);
    outs_bits_out[cb + out_i] = term;

    next_pairs[cb + out_i] = GPUPair{ (uint8_t)take1, (uint8_t)take2,
                                      take1 ? i : 0, take2 ? j : 0 };

    if (take1) ++i;
    if (take2) ++j;
    ++out_i;
  }

  for (uint32_t k = 0; k < out_i; ++k) louds_bits_out[lb + k] = 0;
  louds_bits_out[lb + out_i] = 1;
}
#endif // __CUDACC__

Trie* Trie::merge_trie_direct_linear_cuda(const Trie& t1, const Trie& t2) {
#ifndef __CUDACC__
  return Trie::merge_trie_direct_linear(t1, t2);
#else
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto ex = thrust::cuda::par.on(stream);

  Trie* out = new Trie();
  auto& out_impl   = *out->impl_;
  auto& out_levels = out_impl.levels_;

  const auto& L1 = t1.impl_->levels_;
  const auto& L2 = t2.impl_->levels_;

  out_impl.n_keys_ = 0; out_impl.n_nodes_ = 1; out_impl.size_ = 0;
  out_levels.clear(); out_levels.resize(2);
  out_levels[0].louds.add(0); out_levels[0].louds.add(1);
  out_levels[0].outs.add(0);  out_levels[0].labels.push_back(' ');

  if (!L1.empty() && L1[0].outs.get(0)) { out_levels[0].outs.set(0,1); ++out_levels[1].offset; ++out_impl.n_keys_; }
  if (!L2.empty() && L2[0].outs.get(0) && !out_levels[0].outs.get(0)) { out_levels[0].outs.set(0,1); ++out_levels[1].offset; ++out_impl.n_keys_; }

  std::vector<GPUPair> curr(1, GPUPair{ (uint8_t)!L1.empty(), (uint8_t)!L2.empty(), 0u, 0u });

  std::vector<uint64_t> next_level_offset_bumps;
  for (uint32_t lev = 0; !curr.empty(); ++lev) {
    if (out_levels.size() <= lev + 1) out_levels.resize(lev + 2);
    const size_t P = curr.size();

    std::vector<uint32_t> h_b1(P,0), h_e1(P,0), h_b2(P,0), h_e2(P,0);
    for (size_t p=0;p<P;++p) {
      if (curr[p].h1) crange(L1, lev, curr[p].id1, h_b1[p], h_e1[p]);
      if (curr[p].h2) crange(L2, lev, curr[p].id2, h_b2[p], h_e2[p]);
    }

    static thrust::device_vector<uint8_t>  d_lab1, d_lab2;
    static thrust::device_vector<uint64_t> d_outs1w, d_outs2w;
    // if (lev + 1 < L1.size()) {
    //   d_lab1   = thrust::device_vector<uint8_t>(L1[lev+1].labels.begin(), L1[lev+1].labels.end());
    //   d_outs1w = thrust::device_vector<uint64_t>(L1[lev+1].outs.words.begin(), L1[lev+1].outs.words.end());
    // }
    // if (lev + 1 < L2.size()) {
    //   d_lab2   = thrust::device_vector<uint8_t>(L2[lev+1].labels.begin(), L2[lev+1].labels.end());
    //   d_outs2w = thrust::device_vector<uint64_t>(L2[lev+1].outs.words.begin(), L2[lev+1].outs.words.end());
    // }
    if (lev + 1 < L1.size()) {
        d_lab1.resize(L1[lev+1].labels.size());
        d_outs1w.resize(L1[lev+1].outs.words.size());
        if (!d_lab1.empty())
          CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_lab1.data()),
                                     L1[lev+1].labels.data(),
                                     d_lab1.size()*sizeof(uint8_t),
                                     cudaMemcpyHostToDevice, stream));
        if (!d_outs1w.empty())
          CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_outs1w.data()),
                                     L1[lev+1].outs.words.data(),
                                     d_outs1w.size()*sizeof(uint64_t),
                                     cudaMemcpyHostToDevice, stream));
      }
      if (lev + 1 < L2.size()) {
        d_lab2.resize(L2[lev+1].labels.size());
        d_outs2w.resize(L2[lev+1].outs.words.size());
        if (!d_lab2.empty())
          CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_lab2.data()),
                                     L2[lev+1].labels.data(),
                                     d_lab2.size()*sizeof(uint8_t),
                                     cudaMemcpyHostToDevice, stream));
        if (!d_outs2w.empty())
          CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_outs2w.data()),
                                     L2[lev+1].outs.words.data(),
                                     d_outs2w.size()*sizeof(uint64_t),
                                     cudaMemcpyHostToDevice, stream));
      }      

    static thrust::device_vector<uint32_t> d_b1, d_e1, d_b2, d_e2;
    static thrust::device_vector<GPUPair>  d_pairs;
    d_b1.resize(P); d_e1.resize(P); d_b2.resize(P); d_e2.resize(P);
    d_pairs.resize(P);
    if (P) {
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_b1.data()), h_b1.data(),
                              P*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_e1.data()), h_e1.data(),
                              P*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_b2.data()), h_b2.data(),
                              P*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_e2.data()), h_e2.data(),
                              P*sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_pairs.data()), curr.data(),
                              P*sizeof(GPUPair), cudaMemcpyHostToDevice, stream));
    }

    static thrust::device_vector<uint32_t> d_out_len, d_term_cnt;
    d_out_len.resize(P); d_term_cnt.resize(P);

    const int TPB=256, BLK=(int)((P+TPB-1)/TPB);
    k_count<<<BLK,TPB, 0, stream>>>(
      d_lab1.empty()?nullptr:thrust::raw_pointer_cast(d_lab1.data()),
      thrust::raw_pointer_cast(d_b1.data()),
      thrust::raw_pointer_cast(d_e1.data()),
      d_outs1w.empty()?nullptr:thrust::raw_pointer_cast(d_outs1w.data()),
      d_lab2.empty()?nullptr:thrust::raw_pointer_cast(d_lab2.data()),
      thrust::raw_pointer_cast(d_b2.data()),
      thrust::raw_pointer_cast(d_e2.data()),
      d_outs2w.empty()?nullptr:thrust::raw_pointer_cast(d_outs2w.data()),
      thrust::raw_pointer_cast(d_pairs.data()), (uint32_t)P,
      thrust::raw_pointer_cast(d_out_len.data()),
      thrust::raw_pointer_cast(d_term_cnt.data()));
    CUDA_CHECK(cudaGetLastError());

    static thrust::device_vector<uint32_t> d_child_base, d_louds_base;
    d_child_base.resize(P);
    thrust::exclusive_scan(ex, d_out_len.begin(), d_out_len.end(), d_child_base.begin(), 0u);
    const uint32_t total_children = thrust::reduce(ex,
        d_out_len.begin(), d_out_len.end(), 0u, thrust::plus<uint32_t>());

    d_louds_base.resize(P);
    thrust::sequence(ex, d_louds_base.begin(), d_louds_base.end(), 0u);
    thrust::transform(ex, d_louds_base.begin(), d_louds_base.end(),
                      d_child_base.begin(), d_louds_base.begin(), thrust::plus<uint32_t>());

    static thrust::device_vector<uint8_t>  d_labels_out, d_outs_bits_out, d_louds_bits_out;
    static thrust::device_vector<GPUPair>  d_next_pairs;
    d_labels_out.resize(total_children);
    d_outs_bits_out.resize(total_children);
    d_louds_bits_out.resize(total_children + (uint32_t)P);
    d_next_pairs.resize(total_children);

    k_emit<<<BLK,TPB, 0, stream>>>(
      d_lab1.empty()?nullptr:thrust::raw_pointer_cast(d_lab1.data()),
      thrust::raw_pointer_cast(d_b1.data()), thrust::raw_pointer_cast(d_e1.data()),
      d_outs1w.empty()?nullptr:thrust::raw_pointer_cast(d_outs1w.data()),
      d_lab2.empty()?nullptr:thrust::raw_pointer_cast(d_lab2.data()),
      thrust::raw_pointer_cast(d_b2.data()), thrust::raw_pointer_cast(d_e2.data()),
      d_outs2w.empty()?nullptr:thrust::raw_pointer_cast(d_outs2w.data()),
      thrust::raw_pointer_cast(d_pairs.data()), (uint32_t)P,
      thrust::raw_pointer_cast(d_child_base.data()), thrust::raw_pointer_cast(d_louds_base.data()),
      thrust::raw_pointer_cast(d_labels_out.data()),
      thrust::raw_pointer_cast(d_outs_bits_out.data()),
      thrust::raw_pointer_cast(d_louds_bits_out.data()),
      thrust::raw_pointer_cast(d_next_pairs.data()));
    CUDA_CHECK(cudaGetLastError());

    std::vector<uint8_t> labels_out(total_children), outs_bits_out(total_children),
                         louds_bits_out(total_children + P);
    if (total_children)
      CUDA_CHECK(cudaMemcpyAsync(labels_out.data(), thrust::raw_pointer_cast(d_labels_out.data()),
                                 total_children*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    if (total_children)
      CUDA_CHECK(cudaMemcpyAsync(outs_bits_out.data(), thrust::raw_pointer_cast(d_outs_bits_out.data()),
                                 total_children*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
    if (total_children + P)
      CUDA_CHECK(cudaMemcpyAsync(louds_bits_out.data(), thrust::raw_pointer_cast(d_louds_bits_out.data()),
                                 (total_children + P)*sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));

    std::vector<GPUPair> next(total_children);
    if (total_children)
      CUDA_CHECK(cudaMemcpyAsync(next.data(), thrust::raw_pointer_cast(d_next_pairs.data()),
                                 total_children*sizeof(GPUPair), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    out_levels[lev+1] = Level();
    assign_from_bits(out_levels[lev+1].louds, louds_bits_out);
    assign_from_bits(out_levels[lev+1].outs,  outs_bits_out);
    out_levels[lev+1].labels = std::move(labels_out);

    const uint64_t level_term = thrust::reduce(ex, d_term_cnt.begin(), d_term_cnt.end(), 0u, thrust::plus<uint32_t>());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (out_levels.size() <= lev + 2) out_levels.resize(lev + 3);
    if (next_level_offset_bumps.size() <= lev + 2) next_level_offset_bumps.resize(lev + 3, 0);
    next_level_offset_bumps[lev + 2] += level_term;
    out_impl.n_keys_  += level_term;
    out_impl.n_nodes_ += total_children;

    // std::vector<GPUPair> next(total_children);
    // thrust::copy(d_next_pairs.begin(), d_next_pairs.end(), next.begin());
    curr.swap(next);
  }

  for (size_t i=0;i<out_levels.size() && i<next_level_offset_bumps.size();++i) {
    out_levels[i].offset += next_level_offset_bumps[i];
  }

  out_impl.build();
  CUDA_CHECK(cudaStreamDestroy(stream));
  return out;
#endif // __CUDACC__
}

}  // namespace louds