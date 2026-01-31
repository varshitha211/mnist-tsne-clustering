import streamlit as st
import plotly.express as px

from data_utils import load_mnist
from clustering_utils import apply_tsne, apply_kmeans
from metrics import cluster_stats, clustering_accuracy, silhouette

st.set_page_config(layout="wide")
st.title("MNIST Digit Clustering (Full MNIST + t-SNE)")

# Sidebar controls
st.sidebar.header("Controls")
k = st.sidebar.slider("Number of clusters (k)", 3, 15, 10)
sample_size = st.sidebar.selectbox("Sample size", [5000, 7000, 10000])

# Load data
X, y = load_mnist(sample_size=sample_size)

# t-SNE
with st.spinner("Running t-SNE (this may take up to a minute)..."):
    X_2d = apply_tsne(X)

# Clustering
clusters = apply_kmeans(X_2d, k)

# Metrics
acc = clustering_accuracy(clusters, y)
sil = silhouette(X_2d, clusters)

st.subheader("Clustering Metrics")
st.write(f"Approximate Accuracy: {acc:.2f}")
st.write(f"Silhouette Score: {sil:.2f}")

# Plot
fig = px.scatter(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    color=clusters.astype(str),
    hover_data={"True Digit": y, "Cluster": clusters},
    title="t-SNE Visualization of Full MNIST (Sampled)"
)

st.plotly_chart(fig, use_container_width=True)

# Stats
st.subheader("Cluster Statistics")
st.json(cluster_stats(clusters, y))

# -------------------------------
# Sample Misclassified Digits (LIMITED)
# -------------------------------

st.subheader("Sample Misclassified Digits (Limited)")

# Find misclassified samples
misclassified_idx = []

for i in range(len(y)):
    # Majority label for this cluster
    cluster_label = clusters[i]
    majority_digit = max(
        set(y[clusters == cluster_label]),
        key=list(y[clusters == cluster_label]).count
    )

    if y[i] != majority_digit:
        misclassified_idx.append(i)

# Show only first 5 to avoid overload
misclassified_idx = misclassified_idx[:5]

if len(misclassified_idx) == 0:
    st.write("No misclassified samples found.")
else:
    cols = st.columns(len(misclassified_idx))

    for col, idx in zip(cols, misclassified_idx):
        image = X[idx].reshape(28, 28)
        col.image(
            image,
            caption=f"True: {y[idx]}, Cluster: {clusters[idx]}",
            clamp=True
        )
