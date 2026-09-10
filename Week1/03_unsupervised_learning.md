# Unsupervised Learning

## Definition

Learning from data that has no labeled output — the model has to find structure, patterns, or groupings on its own.

## Common tasks

- **Clustering** — group similar points together (e.g., K-means)
- **Dimensionality reduction** — compress features while keeping important structure (e.g., PCA)
- **Density estimation** — model how data is distributed

## Clustering (visual intuition)

- Slide shows unlabeled points scattered across a plane, forming visually distinct blobs (dense blue group, pink group, green group, and a sparse white group with a couple of stray points in between)
- No colors/labels were given to the model beforehand — the "blob" shape itself is what a clustering algorithm is trying to discover
- **Why it matters**: this is the concrete picture behind "Clustering" above — the algorithm's job is to look at raw, unlabeled (x, y) points and partition them into groups like these ellipses, purely based on how close/similar points are to each other. Stray points between blobs are a preview of why cluster boundaries aren't always clean.

## Applications

- **Customer Data** → discover classes/segments of customers
- **Image pixels** → discover regions (e.g., segmenting a photo into sky, sand, land, tree — shown via a beach image split into flat color regions)
- **Words** → discover synonyms
- **Documents** → discover topics
- **Why it matters**: each of these is "clustering" or "structure discovery" applied to a different data type —