# %% packages
import pandas as pd
from plotnine import ggplot, aes, geom_text, labs
from sklearn.manifold import TSNE
import torchtext.vocab as vocab
import torch

# %% import GloVe
glove_dim = 100
glove = vocab.GloVe(name="6B", dim=glove_dim)


# %% Get closest words from word input
def get_embedding_vector(word):
    word_index = glove.stoi[word]
    emb = glove.vectors[word_index]
    return emb


def get_closest_words_from_word(word, max_n=5):
    word_emb = get_embedding_vector(word)
    distances = [
        (w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item())
        for w in glove.itos
    ]
    dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
    return [item[0] for item in dist_sort_filt]


get_closest_words_from_word(word="chess", max_n=10)

# %%
words = []
categories = ["numbers", "algebra", "music", "science", "technology"]
df_word_cloud = pd.DataFrame(columns=["category", "word"])

categories_df = []
for category in categories:
    words = get_closest_words_from_word(category, max_n=20)
    categories_df.append(pd.DataFrame({"category": category, "word": words}))

df_word_cloud = pd.concat(categories_df, ignore_index=True).reset_index(drop=True)

# %% get embeddings from words
n_rows = len(df_word_cloud)
n_cols = glove_dim
X = torch.empty((n_rows, n_cols))
word_column = df_word_cloud["word"].values
for i, word in enumerate(word_column):
    X[i, :] = get_embedding_vector(word)
    print(f"Word: {word} - {i}/{n_rows}")

# %% t-SNE
tsne = TSNE(n_components=2, perplexity=10)
X_tsne = tsne.fit_transform(X.cpu().numpy())

df_word_cloud["x"] = X_tsne[:, 0]
df_word_cloud["y"] = X_tsne[:, 1]


(
    ggplot(df_word_cloud.sample(25), aes(x="x", y="y", label="word", color="category"))
    + geom_text()
    + labs(title="GloVe Word and category", x="t-SNE 1", y="t-SNE 2")
)
