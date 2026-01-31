# MNIST Digit Clustering with t-SNE

## Overview
This project demonstrates how unsupervised machine learning can be used to discover patterns in handwritten digit images.  
It uses dimensionality reduction (t-SNE) and clustering (k-means) to group visually similar digits and visualize how the system organizes the data.

The goal of this project is not classification, but **understanding and visualizing similarity structure** in high-dimensional image data.

---

## Dataset
This project uses the **MNIST handwritten digits dataset**, which contains **70,000 images** of digits from 0 to 9.  
Each image is **28×28 pixels** and represents a grayscale handwritten digit.

To keep the visualization interactive and computationally feasible, a **random subset of 5,000 images** is sampled from the full dataset.  
This follows the project recommendation to sample 5,000–10,000 images when working with full MNIST.

### Why Sampling?
t-SNE is computationally expensive and designed for exploratory visualization rather than real-time processing of very large datasets.  
Sampling preserves meaningful similarity structure while keeping the application responsive and usable.

---

## Technologies Used
- Python  
- NumPy  
- scikit-learn (t-SNE, k-means, evaluation metrics)  
- Streamlit (interactive interface)  
- Plotly (interactive visualization)

---

## Approach
1. Loaded the MNIST dataset and flattened each image into a numerical feature vector.
2. Normalized pixel values to prepare the data for analysis.
3. Applied **t-SNE** to reduce high-dimensional image data into 2D coordinates for visualization.
4. Applied **k-means clustering** on the 2D embeddings to group similar digits.
5. Built an interactive Streamlit application to explore clusters visually.
6. Evaluated clustering quality using approximate accuracy and silhouette score.
7. Displayed a small sample of misclassified digits to understand clustering limitations.

---

## Results & Observations
- The model forms meaningful clusters for many digits, especially simple shapes like **1** and **0**.
- Digits with similar visual structures (such as **4 and 9**, **3 and 5**) show overlap across clusters.
- Approximate clustering accuracy is **~0.87**.
- Silhouette score of **~0.61** indicates reasonably well-separated clusters.
- Misclassified samples highlight natural ambiguity in handwritten digits.

These results demonstrate both the strengths and limitations of unsupervised clustering on image data.

---

## How to Run the Project

Install dependencies:
pip install -r requirements.txt

Run the application:
python -m streamlit run app.py

The app will open automatically in your browser.

---

## Project Structure

mnist_tsne_full/
├── app.py
├── data_utils.py
├── clustering_utils.py
├── metrics.py
├── requirements.txt
├── README.md

---

## Key Learnings
- High-dimensional image data requires dimensionality reduction for visualization.
- Unsupervised clustering groups data based on similarity, not labels.
- Visualization is critical for understanding and debugging machine learning behavior.
- Performance trade-offs are necessary when working with large datasets.

---

## Limitations & Future Improvements
- t-SNE is computationally expensive and not suitable for very large real-time datasets.
- Results may vary slightly due to the stochastic nature of t-SNE.
- Future improvements could include PCA + t-SNE pipelines or alternative embedding methods such as UMAP.

---

## Time Spent
Approximately 6–7 hours, including experimentation, debugging, and visualization improvements.
