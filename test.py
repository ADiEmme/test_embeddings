from FlagEmbedding import BGEM3FlagModel 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Inizializza il modello
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Frasi da embeddare
sentences = [
    "Call of duty, e' un videogioco di merda, sviluppato da activision?",
    "The Finals, e' un videogioco svuluppato da Embark games",
    "La honda hornet e' una moto versatile",
    "Beach volley e' uno sport estivo molto competitivo e complesso",
    "Brawhalla e' un videogioco dove ci si picchia",
    "La PS5 e' una console scarsa",
    "Il calcio e' uno sport formato da 12 giocatori per squadra"
]

query = " Quale sport estivo mi consigli di provare?"

# Calcola embeddings per le frasi
embeddings = model.encode(sentences)['dense_vecs']

# Calcola embedding per la query
query_embedding = model.encode([query])['dense_vecs']

# Combina tutti gli embeddings
all_embeddings = np.vstack([embeddings, query_embedding])

# Salva la matrice embeddings su .npy
np.save("embeddings.npy", all_embeddings)

# Carica embeddings dal file .npy (opzionale)
loaded_embeddings = np.load('embeddings.npy')

# ----------------------------
# Calcolo cosine similarity
# ----------------------------

# Normalizza embeddings
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)

# Calcola cosine similarity
cosine_similarities = embeddings_norm @ query_embedding_norm.T
cosine_similarities = cosine_similarities.flatten()

# Ordina i risultati in ordine decrescente
sorted_indices = np.argsort(-cosine_similarities)
sorted_similarities = cosine_similarities[sorted_indices]
sorted_sentences = [sentences[i] for i in sorted_indices]

# Stampa risultati
print("Similarità coseno della query rispetto a ciascuna frase:")
for sent, score in zip(sorted_sentences, sorted_similarities):
    print(f"{score:.4f} -> {sent}")

# Riduzione dimensionale con T-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
embeddings_2d = tsne.fit_transform(loaded_embeddings)

# Split embeddings 2D
sentences_2d = embeddings_2d[:-1]         # tutte le frasi
query_2d = embeddings_2d[-1]              # la query

# Plot
plt.figure(figsize=(8,6))

# Plot frasi (punti blu)
plt.scatter(sentences_2d[:,0], sentences_2d[:,1], color='blue', label='Sentences')

# Plot query (punto rosso)
plt.scatter(query_2d[0], query_2d[1], color='red', s=100, label='Query')

# Aggiungi etichette alle frasi
for i, sentence in enumerate(sentences):
    plt.text(sentences_2d[i, 0] + 0.1, sentences_2d[i, 1] + 0.1, sentence, fontsize=9)

# Etichetta per la query
plt.text(query_2d[0] + 0.1, query_2d[1] + 0.1, query, fontsize=10, color='red', fontweight='bold')

plt.title("T-SNE Projection of Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


