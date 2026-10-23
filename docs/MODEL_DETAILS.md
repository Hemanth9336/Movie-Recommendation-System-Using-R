# Model Details - Movie Recommendation System using R (Item-Based Collaborative Filtering)

> **Docs:** [README](../README.md) | [Architecture](ARCHITECTURE.md) | [HLD](HLD.md) | [LLD](LLD.md) | [Model Details](MODEL_DETAILS.md) | [AI Primer](AI_ML_GENAI_PRIMER.md) | [Presentation](presentation/Movie_Recommendation_System_R.pptx)

The project uses recommenderlab's Item-Based Collaborative Filtering. Cosine similarity is also used directly to explore user-user similarity.

## Summary

| Model | Family | Result |
|---|---|---|
| Item-Based Collaborative Filtering (IBCF) | Memory-based collaborative filtering (unsupervised similarity) | Top-10 personalised recommendations for every test user; similarity matrix and recommendation-frequency distributions visualised. |
| Cosine similarity (user-user exploration) | Vector similarity measure | 4 x 4 user-similarity heatmap. |

## Item-Based Collaborative Filtering (IBCF)

**Family:** Memory-based collaborative filtering (unsupervised similarity)

### How it works (plain English)

1) Centre each user's ratings (subtract their average) so harsh and generous raters are comparable. 2) Represent every movie as the vector of ratings it received from all training users. 3) Compute cosine similarity between every pair of movies and keep the **30 most similar** per movie. 4) For a test user, score each unseen movie by the similarity-weighted average of the user's ratings on its neighbours. 5) Return the 10 highest-scoring movies.

```mermaid
flowchart LR
  U[User liked: Toy Story 5*, Shrek 4.5*] --> N[Neighbours of liked movies<br/>via item-item cosine sim]
  N --> C[Candidates: Finding Nemo, Monsters Inc, Up ...]
  C --> S[Score = weighted avg of user's ratings]
  S --> T[Top-10 recommendations]
```

### Hyper-parameters used

| Parameter | Value | Meaning |
|---|---|---|
| method | IBCF | Item-based CF |
| k | 30 | Neighbours per item |
| similarity | Cosine (default) | Angle between rating vectors |
| normalize | center (default) | Subtract user mean |
| n (predict) | 10 | Recommendations per user |

### Result

Top-10 personalised recommendations for every test user; similarity matrix and recommendation-frequency distributions visualised.

### Strengths
- Intuitive and explainable ('because you liked...')
- Item similarities can be pre-computed and cached
- No content metadata required

### Weaknesses
- Cold start for new users/movies
- Popularity bias
- Similarity matrix grows quadratically with catalogue size

## Cosine similarity (user-user exploration)

**Family:** Vector similarity measure

### How it works (plain English)

`cos(a, b) = (a . b) / (|a| |b|)`. Used on the first 4 users to visualise a user-similarity heatmap, and internally by IBCF for items. This is exactly the similarity measure used by vector databases to compare text or image embeddings.

### Hyper-parameters used

| Parameter | Value | Meaning |
|---|---|---|
| which | users | Compare rows |
| method | cosine | Similarity measure |

### Result

4 x 4 user-similarity heatmap.

### Strengths
- Scale-independent
- Efficient on sparse data

### Weaknesses
- Ignores rating-count reliability unless combined with thresholds

## Evaluation metrics explained

| Metric | Meaning |
|---|---|
| Top-N list | Ranked list of N recommended items per user (here N = 10). |
| Precision@N / Recall@N (future) | Share of recommended items the user actually liked / share of liked items that were recommended. |
| RMSE (future) | Error between predicted and actual ratings. |
| Coverage / popularity | How many distinct movies get recommended (see item-frequency distribution chart). |

## Full results

| Item | Value |
|---|---|
| Ratings / users / movies | 105,339 / 668 / 10,325 (10,329 titles in catalogue) |
| Rating scale | 0.5 - 5.0 stars, mean 3.52 |
| Matrix sparsity | 98.5% of user-movie cells empty |
| Modelling subset | Users with >50 ratings and movies with >50 ratings (~420 x ~447) |
| Train / test | ~80% / ~20% of users (random sample) |
| Model | IBCF, cosine similarity, k = 30 neighbours per item |
| Output | Top-10 recommended movies per test user + item-frequency distribution |
| Most-viewed films | Pulp Fiction (325), Forrest Gump (311), Shawshank Redemption (308), Jurassic Park (294), Silence of the Lambs (290), Star Wars IV (273) |

- The original project focuses on producing recommendations and visual analysis; it does not compute offline accuracy metrics. `recommenderlab::evaluationScheme` + `evaluate()` can add RMSE / precision@10 / recall@10 (see next steps).
- Filtering to active users and popular movies reduces sparsity and noise - a common, pragmatic choice that also introduces popularity bias.
