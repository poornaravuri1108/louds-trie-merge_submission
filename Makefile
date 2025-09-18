ALL:
	g++ -O2 -Wall -Wextra -march=native *.cpp -o louds-trie

MERGE:
	g++ -O2 -Wall -Wextra -march=native test.cpp louds-trie.cpp -o merge
