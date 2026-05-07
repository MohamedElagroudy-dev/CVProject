"""
Image Similarity Search Engine
================================
Uses SIFT + Bag of Visual Words + TF-IDF + Cosine Similarity
Optimized for slow laptops: small dataset, low K, grayscale only
"""

import os
import cv2
import numpy as np
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Import the visualization module
import show_results


# ─────────────────────────────────────────
#  CONFIG  ← change these to fit your setup
# ─────────────────────────────────────────
DATASET_PATH = "dataset"        # folder that contains your images
SAVE_PATH    = "model"          # folder to save computed data
K            = 2000             # number of visual words (keep low for slow PC)
MAX_IMAGES   = 1000             # max images to load (keep low for slow PC)
#IMAGE_SIZE = (512, 512)  
IMAGE_SIZE   = (256, 256)     # resize all images to this before processing
TOP_N        = 5                # number of search results to show


# ─────────────────────────────────────────
#  STEP 1 — Load images
# ─────────────────────────────────────────
def load_images(dataset_path, max_images=MAX_IMAGES, size=IMAGE_SIZE):
    print("\n[1] Loading images...")
    supported = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = []
    images = []

    for fname in sorted(os.listdir(dataset_path)):
        if fname.lower().endswith(supported):
            image_paths.append(os.path.join(dataset_path, fname))
        if len(image_paths) >= max_images:
            break

    if not image_paths:
        raise FileNotFoundError(f"No images found in '{dataset_path}'")

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # grayscale only
        if img is None:
            print(f"  Warning: could not read {path}, skipping.")
            continue
        img = cv2.resize(img, size)
        # Gaussian blur to reduce noise before SIFT
        img = cv2.GaussianBlur(img, (3, 3), 0)
        images.append(img)

    print(f"  Loaded {len(images)} images.")
    return images, image_paths[:len(images)]


# ─────────────────────────────────────────
#  STEP 2 — Extract SIFT descriptors
# ─────────────────────────────────────────
def extract_sift_descriptors(images):
    print("\n[2] Extracting SIFT descriptors...")
    sift = cv2.SIFT_create(
    nfeatures=1000,
    nOctaveLayers=5,
    contrastThreshold=0.02,
    edgeThreshold=15,
    sigma=1.2
)

    all_descriptors = []   # one flat list of all descriptors (for K-Means)
    per_image_desc  = []   # list of descriptor arrays, one per image

    for i, img in enumerate(images):
        keypoints, descriptors = sift.detectAndCompute(img, None)

        if descriptors is None:
            per_image_desc.append(None)
            print(f"  Image {i+1}: no keypoints found.")
        else:
            all_descriptors.append(descriptors)
            per_image_desc.append(descriptors)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1} / {len(images)} images...")

    all_descriptors = np.vstack(all_descriptors)
    print(f"  Total descriptors collected: {all_descriptors.shape[0]}")
    return all_descriptors, per_image_desc


# ─────────────────────────────────────────
#  STEP 3 — Build visual vocabulary (K-Means)
# ─────────────────────────────────────────
def build_vocabulary(all_descriptors, k=K):
    print(f"\n[3] Building visual vocabulary with K={k} words...")
    print("  This may take a minute on a slow PC...")

    # MiniBatchKMeans is much faster than regular KMeans
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=1000,
        n_init=3,
        max_iter=100,
        verbose=0
    )
    kmeans.fit(all_descriptors)
    print("  Vocabulary built.")
    return kmeans


# ─────────────────────────────────────────
#  STEP 4 — Build histograms (BoVW)
# ─────────────────────────────────────────
def build_histograms(per_image_desc, kmeans, k=K):
    print("\n[4] Building image histograms...")
    histograms = []

    for desc in per_image_desc:
        if desc is None:
            # image had no keypoints — empty histogram
            histograms.append(np.zeros(k))
        else:
            labels = kmeans.predict(desc)
            hist, _ = np.histogram(labels, bins=np.arange(k + 1))
            histograms.append(hist.astype(float))

    histograms = np.array(histograms)
    print(f"  Histogram matrix shape: {histograms.shape}")
    return histograms


# ─────────────────────────────────────────
#  STEP 5 — Apply TF-IDF weighting
# ─────────────────────────────────────────
def apply_tfidf(histograms):
    print("\n[5] Applying TF-IDF weighting...")
    n = histograms.shape[0]  # number of images

    # TF: normalize each row so word frequency is relative
    tf = histograms / (histograms.sum(axis=1, keepdims=True) + 1e-7)

    # IDF: penalize words that appear in many images
    df = (histograms > 0).sum(axis=0)           # document frequency per word
    idf = np.log((n + 1) / (df + 1)) + 1        # smoothed IDF

    tfidf = tf * idf

    # L2-normalize each row so cosine similarity works correctly
    tfidf = normalize(tfidf, norm="l2")

    print("  TF-IDF applied and vectors normalized.")
    return tfidf, idf


# ─────────────────────────────────────────
#  STEP 6 — Search: find similar images
# ─────────────────────────────────────────
def search(query_path, images, image_paths, kmeans, tfidf_matrix, idf,
           top_n=TOP_N, size=IMAGE_SIZE):
    print(f"\n[6] Searching for images similar to: {query_path}")

    # Load and preprocess the query image
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        raise FileNotFoundError(f"Cannot read query image: {query_path}")
    query_img = cv2.resize(query_img, size)
    query_img = cv2.GaussianBlur(query_img, (3, 3), 0)

    # Extract SIFT from query
    sift = cv2.SIFT_create(nfeatures=300)
    _, desc = sift.detectAndCompute(query_img, None)

    if desc is None:
        print("  No keypoints found in query image.")
        return []

    # Build query histogram
    k = tfidf_matrix.shape[1]
    labels = kmeans.predict(desc)
    hist, _ = np.histogram(labels, bins=np.arange(k + 1))
    hist = hist.astype(float)

    # Apply same TF-IDF transform to query
    tf = hist / (hist.sum() + 1e-7)
    query_tfidf = tf * idf
    query_tfidf = query_tfidf / (np.linalg.norm(query_tfidf) + 1e-7)
    query_tfidf = query_tfidf.reshape(1, -1)

    # Cosine similarity against all images
    scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

    # Sort by similarity (highest first)
    ranked = np.argsort(scores)[::-1][:top_n]

    print(f"\n  Top {top_n} similar images:")
    results = []
    for rank, idx in enumerate(ranked):
        fname = os.path.basename(image_paths[idx])
        score = scores[idx]
        print(f"    {rank+1}. {fname}  (score: {score:.4f})")
        results.append((image_paths[idx], score))

    return results


# ─────────────────────────────────────────
#  STEP 7 — Save / Load model
# ─────────────────────────────────────────
def save_model(kmeans, tfidf_matrix, idf, image_paths, save_path=SAVE_PATH):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "model.pkl"), "wb") as f:
        pickle.dump({
            "kmeans":       kmeans,
            "tfidf_matrix": tfidf_matrix,
            "idf":          idf,
            "image_paths":  image_paths
        }, f)
    print(f"\n  Model saved to '{save_path}/model.pkl'")


def load_model(save_path=SAVE_PATH):
    model_file = os.path.join(save_path, "model.pkl")
    if not os.path.exists(model_file):
        return None
    with open(model_file, "rb") as f:
        data = pickle.load(f)
    print(f"\n  Model loaded from '{model_file}'")
    return data


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def build_index():
    """Run the full pipeline and save the model."""
    images, image_paths    = load_images(DATASET_PATH)
    all_desc, per_img_desc = extract_sift_descriptors(images)
    kmeans                 = build_vocabulary(all_desc, k=K)
    histograms             = build_histograms(per_img_desc, kmeans, k=K)
    tfidf_matrix, idf      = apply_tfidf(histograms)

    save_model(kmeans, tfidf_matrix, idf, image_paths)
    return kmeans, tfidf_matrix, idf, image_paths, images


def search_and_display(query_path):
    """Search for similar images and display results visually."""
    data = load_model()
    if data is None:
        print("No saved model found. Run with 'build' first.")
        return
    
    # Load images for display (needed for visualization context)
    images, _ = load_images(DATASET_PATH)
    
    # Perform search
    results = search(
        query_path,
        images,
        data["image_paths"],
        data["kmeans"],
        data["tfidf_matrix"],
        data["idf"],
        top_n=TOP_N
    )
    
    if results:
        # Display results using the imported visualization module
        show_results.display_results(query_path, results, top_n=TOP_N)
    else:
        print("No results found.")


if __name__ == "__main__":
    import sys

    # ── Build mode: python image_similarity_search.py build
    if len(sys.argv) == 2 and sys.argv[1] == "build":
        build_index()
        print("\nDone! Now run:  python image_similarity_search.py search <your_image.jpg>")

    # ── Search mode: python image_similarity_search.py search path/to/query.jpg
    elif len(sys.argv) == 3 and sys.argv[1] == "search":
        search_and_display(sys.argv[2])

    else:
        print("Usage:")
        print("  Build index:  python image_similarity_search.py build")
        print("  Search:       python image_similarity_search.py search path/to/image.jpg")