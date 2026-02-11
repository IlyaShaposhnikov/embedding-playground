**Semantic vector playground: compare Word2Vec vs GloVe on word analogies and nearest neighbors.**

## Goal
- Load pre-trained **Word2Vec (GoogleNews)** and **GloVe (Stanford)** models.
- Implement robust functions for nearest neighbors and word analogies.
- Evaluate models on classic analogy test sets.
- Visualize embeddings (PCA/t‑SNE).

## Quick Start

```bash
git clone https://github.com/IlyaShaposhnikov/embedding-playground.git
cd embedding-playground
python -m venv .venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Features
* Automatic download & verification of Word2Vec / GloVe
* Nearest neighbors with similarity bars
* Word analogy solver (king - man + woman = ?)
* Batch analogy testing & accuracy report
* Model comparison side‑by‑side
* 2D/3D projection of word vectors

## Resources
* [Word2Vec GoogleNews](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view)
* [GloVe by Stanford NLP](https://nlp.stanford.edu/projects/glove/)
* [Gensim documentation](https://radimrehurek.com/gensim/)

## Author

Ilya Shaposhnikov

ilia.a.shaposhnikov@gmail.com