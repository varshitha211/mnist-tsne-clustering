from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def apply_tsne(X):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=1000,
        random_state=42
    )
    return tsne.fit_transform(X)

def apply_kmeans(X_2d, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    return kmeans.fit_predict(X_2d)
