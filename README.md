# NLP_genre_detection

## Goal

The goal of this project was to create a model able to detect the movie genre, given its synopsis

## Task
Multi-Label Text Classification

## Technical Solutions

1. LSTM Network
  - Input Layer
  - Embedding Layer
  - LSTM Layer
  - Linear Layer

2. Pretrained LSTM network
  - Input Layer
  - Embedding Layer (Pretrained --> Glove 300d)
  - LSTM Layer
  - Linear Layer

3. Transformer Network
  - Input Layer
  - Embedding Layer
  - Encoder

## Evaluation

Mean Average Precision at K for the top 5 predicted genres

## Results

| Model                   | Score |
|-------------------------|-------|
| LSTM Network            | 0.55  |
| Pretrained LSTM network | 0.44  |
| Transformer network     | ....  |
