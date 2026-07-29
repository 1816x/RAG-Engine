# Embeddings and cosine distance

An embedding is a vector of numbers that represents a piece of text, an image, or other data, positioned so that semantically similar items land near each other. Retrieval systems embed both the stored documents and the incoming query into the same space, then find the stored vectors nearest the query vector.

Cosine distance measures the angle between two vectors, ignoring their magnitude. Two vectors that point in the same direction have a cosine distance of zero regardless of how long they are; two perpendicular vectors have a cosine distance of one. This scale invariance is why cosine is the common choice for text embeddings, where direction carries the meaning and length often just reflects document length.

A vector index computes cosine distance efficiently by normalizing every vector to unit length when it is inserted. After normalization, cosine distance is simply one minus the dot product, which is cheaper to compute than the full cosine formula.
