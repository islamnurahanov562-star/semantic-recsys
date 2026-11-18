# Mathematical Background

## Text Embeddings

Let $x$ be an item or query string. The SentenceTransformer encoder maps it to a dense vector:

$$
f(x) \in \mathbb{R}^{d}.
$$

In this project, the selected model produces $d=384$ dimensions and the item vectors are stored as float32 values.

## Cosine Similarity and Inner Product

For vectors $x$ and $y$, cosine similarity is

$$
\operatorname{cos}(x,y)=\frac{x^\top y}{\lVert x\rVert_2\lVert y\rVert_2}.
$$

If both vectors are L2-normalized, then $\lVert x\rVert_2=\lVert y\rVert_2=1$. Therefore,

$$
\operatorname{cos}(x,y)=\frac{x^\top y}{1\cdot1}=x^\top y.
$$

This is why the indexes use inner product for the normalized vectors.

## Exact Top-k Retrieval

For an item matrix $V\in\mathbb{R}^{n\times d}$ and query matrix $Q\in\mathbb{R}^{m\times d}$, all pairwise scores are

$$
S=QV^\top, \qquad S\in\mathbb{R}^{m\times n}.
$$

For one query, exact retrieval ranks item IDs by descending score and returns the first $k$ IDs:

$$
\operatorname{topk}(q)=\operatorname*{arg\,topk}_{i\in\{0,\ldots,n-1\}} q^\top v_i.
$$

The exact computation costs $O(nd)$ operations per query and stores $nd$ float32 values. For $n=8{,}804$ items and $d=384$, the raw vector matrix uses

$$
8{,}804\cdot384\cdot4=13{,}522{,}944\text{ bytes}=12.90\text{ MiB}.
$$

The `.npy` artifact is slightly larger because it also contains a file header.

## IVF Approximate Search

FAISS IVF partitions the vector space into `nlist` coarse clusters and assigns vectors to inverted lists. For each query it probes `nprobe` lists, then ranks vectors found in those lists. This reduces the number of item vectors compared at query time but can omit a true nearest neighbor that resides in an unprobed list.

A rough search-work expression is

$$
O\left(n_{list}d+n_{probe}\frac{n}{n_{list}}d\right).
$$

The trade-off is controlled by `nprobe`: probing more lists usually improves recall and increases latency. The current factory uses the FAISS IVF defaults `nlist=100` and `nprobe=10` for the 8,804-item catalog.

## Retrieval Metrics

For a query with relevant set $R$ and retrieved top-$k$ set $T_k$,

$$
\operatorname{Recall@k}=\frac{|R\cap T_k|}{|R|}.
$$

The reported Recall@k is the mean over queries. Mean reciprocal rank uses the rank of the first relevant result within the top ten:

$$
\operatorname{MRR}=\frac{1}{N}\sum_{j=1}^{N}\begin{cases}\frac{1}{r_j},&\text{if a relevant result occurs at rank }r_j\le10,\\0,&\text{otherwise.}\end{cases}
$$

## Worked Example

Use $q=[1,2,2]$ as a query and the three candidate vectors $v_1=[1,2,2]$, $v_2=[2,0,1]$, and $v_3=[0,3,4]$. The query norm is $\lVert q\rVert_2=3$. The computed dot products, candidate norms, cosine similarities, and dot products after normalization are:

| Candidate | $\lVert v_i\rVert_2$ | $q^\top v_i$ | Cosine similarity | Normalized inner product |
| --- | ---: | ---: | ---: | ---: |
| $v_1=[1,2,2]$ | 3.0000 | 9.0000 | 1.0000 | 1.0000 |
| $v_2=[2,0,1]$ | 2.2361 | 4.0000 | 0.5963 | 0.5963 |
| $v_3=[0,3,4]$ | 5.0000 | 14.0000 | 0.9333 | 0.9333 |

The raw dot products rank $v_3$ above $v_1$ because the vectors have different lengths. After normalization, the inner products equal cosine similarity and correctly rank $v_1$ first for this query. The values were calculated directly with NumPy in float64 arithmetic.

## References

- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*.
- Johnson, J., Douze, A., & Jégou, H. (2017). *Billion-scale similarity search with GPUs*.
