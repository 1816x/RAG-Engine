# How HNSW works

HNSW stands for Hierarchical Navigable Small World. It is an algorithm for approximate nearest neighbor search over high-dimensional vectors.

The core idea is a layered graph. Every vector is a node in the bottom layer, layer zero. Each node is also promoted to higher layers with a probability that decays exponentially, so the top layers are sparse and the bottom layer is dense. This mirrors a skip list, but in the space of vectors instead of a sorted line.

A search starts at the single entry point in the topmost layer and greedily walks toward the query, hopping to whichever neighbor is closest, until it can get no closer. It then drops down a layer and repeats. The sparse top layers make long jumps across the space cheaply; the dense bottom layer refines the result. On the bottom layer a beam search of width ef collects the k nearest neighbors.

The quality of the graph depends on how neighbors are chosen when a node is inserted. A naive choice keeps the M closest candidates. HNSW instead uses a diversity heuristic: a candidate is kept only if it is closer to the new node than to any already-selected neighbor. This spreads links across directions and keeps the graph navigable through sparse regions.
