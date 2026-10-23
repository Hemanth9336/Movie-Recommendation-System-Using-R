# Architecture - Movie Recommendation System using R (Item-Based Collaborative Filtering)

> **Docs:** [README](../README.md) | [Architecture](ARCHITECTURE.md) | [HLD](HLD.md) | [LLD](LLD.md) | [Model Details](MODEL_DETAILS.md) | [AI Primer](AI_ML_GENAI_PRIMER.md) | [Presentation](presentation/Movie_Recommendation_System_R.pptx)

## 1. System context

A single R script that loads MovieLens CSVs, builds a sparse rating matrix, explores it visually, trains an item-based collaborative filtering model on active users / popular movies, and generates top-10 recommendations for held-out users.

## 2. Architecture diagram

```mermaid
flowchart LR
  A[(movies.csv<br/>10,329 titles + genres)] --> G[Genre one-hot matrix<br/>18 genres]
  B[(ratings.csv<br/>105,339 ratings)] --> C[dcast -> user x movie matrix]
  C --> D[realRatingMatrix<br/>668 x 10,325]
  D --> E[EDA: views, heatmaps,<br/>avg rating, normalisation]
  D --> F[Filter users & movies<br/>with > 50 ratings]
  F --> H[Random 80/20 user split]
  H --> I[Recommender IBCF<br/>cosine, k = 30]
  I --> J[Item-item similarity matrix]
  H --> K[Test users]
  J --> L[predict n = 10]
  K --> L
  L --> M[Top-10 movie titles per user]
  G --> S[search_matrix<br/>movieId + title + genres]
```

## 3. Layered view

| Layer | Responsibility | Implementation |
|---|---|---|
| Data layer | Movie catalogue and ratings | `movies.csv`, `ratings.csv` |
| Transformation layer | Genre one-hot matrix; ratings pivot to user x movie | data.table `tstrsplit`, reshape2 `dcast` |
| Matrix layer | Sparse `realRatingMatrix` | recommenderlab |
| Analysis layer | View counts, heatmaps, average ratings, normalisation, binarisation | ggplot2, `image()` |
| Model layer | IBCF with cosine similarity, k = 30 | recommenderlab `Recommender` |
| Recommendation layer | Top-10 per user, map IDs to titles, item frequency | `predict`, base R |

## 4. End-to-end data flow

```mermaid
sequenceDiagram
  participant A as Analyst (RStudio)
  participant R as R script
  participant RL as recommenderlab
  A->>R: run script
  R->>R: read movies.csv, ratings.csv
  R->>R: tstrsplit genres -> 18-col genre matrix
  R->>R: dcast(userId ~ movieId, rating)
  R->>RL: as(matrix, "realRatingMatrix")
  R->>RL: similarity(users 1-4, cosine), colCounts, normalize, binarize
  R->>R: keep rowCounts > 50 & colCounts > 50
  R->>R: sample 80/20 users
  R->>RL: Recommender(train, "IBCF", k = 30)
  RL-->>R: item-item similarity model
  R->>RL: predict(model, test, n = 10)
  RL-->>R: top-10 item IDs per user
  R-->>A: titles for user 1, recommendation matrix, distribution plots
```

## 5. Deployment view

Runs locally in R/RStudio; all data fits in memory. For serving, the model could be wrapped with `plumber` (R REST API) or ported to Python.

## 6. Target (production / GenAI-ready) architecture

A modern, GenAI-enabled recommender built on the same ideas:

```mermaid
flowchart LR
  U[User chat: 'feel-good 90s movie'] --> LLM[LLM + prompt]
  H[(Watch history)] --> CF[CF / embedding model]
  CF --> V[(Vector DB of movie embeddings)]
  LLM -->|query embedding| V
  V --> C[Top-k candidate movies]
  C --> LLM
  LLM --> R[Ranked, explained recommendations]
```
