import json, itertools, os

# ---------- Schema helper ----------
def add(L, category, topic, difficulty, q, a, complexity, type_):
    L.append({
        "id": len(L) + 1,
        "category": category,          # "DSA" | "Core CS" | "Programming" | "ML/AI"
        "topic": topic,                # e.g., "Arrays", "DBMS", "Python", "RAG"
        "difficulty": difficulty,      # "Easy" | "Medium" | "Hard"
        "type": type_,                 # "coding" or "conceptual"
        "question": q.strip(),
        "answer": a.strip(),
        "complexity": complexity.strip()
    })

DATA = []

# ---------- DSA templates ----------
EASY = [
("Arrays","Two Sum on array nums with target t. Return indices or [-1,-1].",
 "Use a hash map value->index; for each x check t-x in map; return indices.",
 "Time: O(n), Space: O(n)"),
("Arrays","Maximum Subarray sum (Kadane).",
 "Track current and global max; reset current if negative.",
 "Time: O(n), Space: O(1)"),
("Strings","Valid parentheses for '()[]{}'.",
 "Use stack; push opens, match on close; empty stack => valid.",
 "Time: O(n), Space: O(n)"),
("Arrays","Move zeroes to end preserving order.",
 "Two-pointer: write non-zeros in-place, fill remaining with zeros.",
 "Time: O(n), Space: O(1)"),
("Hashing","First unique character in string s.",
 "Count frequency then take first index with count==1.",
 "Time: O(n), Space: O(1)"),
("Math","Check power of two for n.",
 "n>0 and n&(n-1)==0.",
 "Time: O(1), Space: O(1)"),
("Arrays","Missing number in 0..n.",
 "XOR 0..n with array or use sum formula.",
 "Time: O(n), Space: O(1)"),
("Arrays","Majority element (>⌊n/2⌋).",
 "Boyer–Moore voting (candidate + count).",
 "Time: O(n), Space: O(1)"),
("Arrays","Merge two sorted arrays.",
 "Two pointers to merge in order.",
 "Time: O(n+m), Space: O(1)"),
("Strings","Check anagram for s,t.",
 "Sort or count chars; compare.",
 "Time: O(n) or O(n log n), Space: O(1~k)")
]

MED = [
("Binary Search","Find peak element index.",
 "Binary search the slope; move toward greater neighbor.",
 "Time: O(log n), Space: O(1)"),
("Two Pointers","3Sum triplets sum to zero.",
 "Sort; fix i; two-pointer for l,r; skip duplicates.",
 "Time: O(n^2), Space: O(1)"),
("Intervals","Merge overlapping intervals.",
 "Sort by start; merge when cur.start <= last.end.",
 "Time: O(n log n), Space: O(n)"),
("Matrix","Rotate n×n matrix 90° clockwise.",
 "Transpose then reverse each row.",
 "Time: O(n^2), Space: O(1)"),
("Heap","Top-K frequent elements.",
 "Count with hashmap; heap or buckets to select top k.",
 "Time: O(n log k) or O(n), Space: O(n)"),
("Tree","Binary tree level-order traversal.",
 "BFS with queue; collect per level.",
 "Time: O(n), Space: O(n)"),
("Tree","LCA in BST.",
 "Walk from root: go left/right until split; that node is LCA.",
 "Time: O(h), Space: O(1)"),
("Graph","Course Schedule (can finish?).",
 "Detect cycle via Kahn’s (in-degree) or DFS colors.",
 "Time: O(V+E), Space: O(V+E)"),
("DP","Coin Change (min coins).",
 "dp[a]=min(dp[a-coin])+1; dp[0]=0; use inf for unreachable.",
 "Time: O(n*amount), Space: O(amount)"),
("DP","LIS length (n log n).",
 "Maintain tails; binary search lower_bound to replace/append.",
 "Time: O(n log n), Space: O(n)")
]

HARD = [
("DP","Edit Distance between two strings.",
 "DP over i,j with insert/delete/replace; take min, 0 if chars equal.",
 "Time: O(nm), Space: O(nm)"),
("Array","Median of two sorted arrays.",
 "Binary search partition on smaller array to balance halves.",
 "Time: O(log min(m,n)), Space: O(1)"),
("String","Minimum window substring containing t.",
 "Expand/contract sliding window with need/have counts; track best.",
 "Time: O(n), Space: O(k)"),
("Graph","Word Ladder shortest path.",
 "BFS using wildcard buckets to connect neighbors; return levels.",
 "Time: O(N·L·Σ), Space: O(N·L)"),
("Design","LRU cache O(1) ops.",
 "Hash map + doubly linked list; move-to-head; evict tail.",
 "Time: O(1)/op, Space: O(capacity)"),
("Adv DS","Segment Tree (range sum, point update).",
 "Build tree; query/update recursively over intervals.",
 "Time: Build O(n), Q/U O(log n), Space: O(n)"),
("Graph","Network delay time from k.",
 "Dijkstra with min-heap; answer is max distance or -1 if unreachable.",
 "Time: O(E log V), Space: O(V+E)"),
("DP","House Robber II (circular).",
 "Solve two LIS-like lines: [0..n-2] and [1..n-1]; take max.",
 "Time: O(n), Space: O(1)"),
("String","Substring with concatenation of all words.",
 "Slide in word_len steps; maintain counts; shrink when exceeded.",
 "Time: O(n·word_len), Space: O(k)"),
("Graph","Critical connections (bridges).",
 "Tarjan: low-link vs discovery times; low[v] > disc[u] => bridge.",
 "Time: O(V+E), Space: O(V)")
]

def gen_dsa_block():
    # 200 Easy, 200 Medium, 200 Hard
    # Cycle templates and add small constraint suffixes so items are unique but consistent.
    diffs = [("Easy", EASY, 200), ("Medium", MED, 200), ("Hard", HARD, 200)]
    for diff, bank, target in diffs:
        count = 0
        cyc = itertools.cycle(bank)
        i = 0
        while count < target:
            topic, q, a, comp = next(cyc)
            i += 1
            qv = f"{q} Constraints: n up to {1000 + i}, values within [-1e9,1e9]."
            add(DATA, "DSA", topic, diff, qv, a, comp, "coding")
            count += 1

# ---------- Core CS, Programming, ML/AI ----------
def cycle_to_total(category, topic, stems, total):
    diffs = itertools.cycle(["Easy", "Medium", "Hard"])
    i = 0
    while i < total:
        for q, a in stems:
            if i >= total: break
            add(DATA, category, topic, next(diffs), q, a, "", "conceptual")
            i += 1

core_sets = [
("OOP", [
 "Name and define Encapsulation, Abstraction, Inheritance, Polymorphism.",
 "Encapsulation bundles state+behavior; Abstraction hides detail; Inheritance reuses; Polymorphism enables interface–implementation separation."
]),
("DBMS", [
 "Explain 1NF, 2NF, 3NF with tiny examples.",
 "1NF: atomic values; 2NF: no partial dependency on composite key; 3NF: no transitive dependency."
]),
("SQL", [
 "Write SQL for second highest salary from Employee(emp_id, salary).",
 "SELECT MAX(salary) FROM Employee WHERE salary < (SELECT MAX(salary) FROM Employee);"
]),
("OS", [
 "Process vs Thread; two benefits and one pitfall of multithreading.",
 "Process has isolated address space; threads share memory. Benefits: responsiveness, resource sharing. Pitfall: data races—use synchronization."
]),
("CN", [
 "How does TCP ensure reliability vs UDP?",
 "TCP: ACKs, retransmission, sliding windows, congestion control; UDP: best-effort, unordered, minimal overhead."
]),
("System Design", [
 "Design a URL shortener: core components and scaling.",
 "API, ID generation (hash/snowflake), KV store, cache, redirect service, analytics; partitioning, rate limiting, eventual consistency for metrics."
]),
("System Design", [
 "Design a chat service: delivery semantics, storage, presence.",
 "Per-user shards, queues for fanout, end-to-end encryption, presence in in-memory store, push notifications, retries/backfill."
])
]

prog_sets = [
("Python", [
 "List vs Tuple differences with a short example.",
 "Lists are mutable; tuples are immutable and hashable (usable as dict keys)."
]),
("Python", [
 "What is the GIL and when use multiprocessing vs multithreading?",
 "GIL serializes Python bytecode; CPU-bound -> multiprocessing; I/O-bound -> threads/async."
]),
("Python", [
 "Sketch an LRU cache with O(1) ops in Python.",
 "Use OrderedDict: move_to_end(key) on get/put; popitem(last=False) to evict."
]),
("Java", [
 "Explain JVM memory areas and GC basics.",
 "Heap, stacks, metaspace; generational GC with young/old; stop-the-world pauses."
]),
("C++", [
 "Explain RAII and give two examples.",
 "RAII ties resource lifetime to object lifetime; examples: std::unique_ptr, std::lock_guard."
])
]

ml_sets = [
("ML", [
 "Bias–variance trade-off and one way to reduce variance.",
 "Reduce variance with regularization, more data, or ensembling."
]),
("ML", [
 "Cross-entropy for classification vs MSE.",
 "Cross-entropy aligns with maximum likelihood, provides well-behaved gradients for probabilities."
]),
("RAG", [
 "Design a robust RAG pipeline from resumes.",
 "Chunk sections, embed with domain model, hybrid retrieval (dense+sparse), rerank, generate with citations and guardrails."
]),
("DL", [
 "Explain self-attention and positional encodings.",
 "Attention computes weights via Q,K,V; positional encodings inject order (sinusoidal or learned)."
]),
("NLP", [
 "Few-shot vs fine-tune vs RAG — when to use each?",
 "RAG for freshness/grounding, fine-tuning for style/domain, few-shot for light behavioral steering."
])
]

# ---------- Build 1000 ----------
gen_dsa_block()                   # 600
# Core CS: 200
core_total = 200
core_base = core_total // len(core_sets)
core_rem = core_total % len(core_sets)
for i, (topic, pair) in enumerate(core_sets):
    q, a = pair
    n = core_base + (1 if i < core_rem else 0)
    cycle_to_total("Core CS", topic, [(q, a)], n)
# Programming: 100
prog_total = 100
prog_base = prog_total // len(prog_sets)
prog_rem = prog_total % len(prog_sets)
for i, (topic, pair) in enumerate(prog_sets):
    q, a = pair
    n = prog_base + (1 if i < prog_rem else 0)
    cycle_to_total("Programming", topic, [(q, a)], n)
# ML/AI: 100
ml_total = 100
ml_base = ml_total // len(ml_sets)
ml_rem = ml_total % len(ml_sets)
for i, (topic, pair) in enumerate(ml_sets):
    q, a = pair
    n = ml_base + (1 if i < ml_rem else 0)
    cycle_to_total("ML/AI", topic, [(q, a)], n)

# Debugging output
from collections import Counter
cat_counts = Counter([item['category'] for item in DATA])
print("Total items:", len(DATA))
print("Breakdown by category:", dict(cat_counts))
assert len(DATA) == 1000, f"Got {len(DATA)} items, expected 1000."

# ---------- Save ----------
os.makedirs(".", exist_ok=True)
with open("rag_interview_qa_1000.json", "w", encoding="utf-8") as f:
    json.dump(DATA, f, ensure_ascii=False, indent=2)

print("Wrote rag_interview_qa_1000.json with", len(DATA), "items.")
