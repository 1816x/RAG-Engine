# Brute-force search and its trade-offs

Brute-force nearest neighbor search compares the query vector against every vector in the collection and returns the closest ones. It is exact by construction: it always returns the true nearest neighbors, because it examines all of them.

The cost is linear in the number of vectors. For a collection of one million vectors, every query touches all one million. This is fine for small collections or infrequent queries, but it does not scale to large corpora under interactive latency requirements.

Brute force is the correct baseline to measure an approximate index against. Because it is exact, its results are the ground truth for recall: recall at k is the fraction of the true k nearest neighbors that the approximate index also returned. It is also the latency baseline — an approximate index is only worthwhile if it is meaningfully faster than scanning everything.
