import os, cv2, pickle, numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import show_results

# ─────────────────────
# CONFIG
# ─────────────────────
DATASET_PATH = "dataset"
SAVE_PATH    = "model"

K            = 2000
MAX_IMAGES   = 1000
IMAGE_SIZE   = (256,256)
TOP_N        = 5


# ─────────────────────
# LOAD IMAGES
# ─────────────────────
def load_images():

    paths, imgs = [], []

    for f in sorted(os.listdir(DATASET_PATH)):

        if f.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            paths.append(os.path.join(DATASET_PATH,f))

        if len(paths) >= MAX_IMAGES:
            break

    for p in paths:

        img = cv2.imread(p,0)

        if img is not None:
            imgs.append(cv2.resize(img, IMAGE_SIZE))

    return imgs, paths[:len(imgs)]


# ─────────────────────
# EXTRACT SIFT
# ─────────────────────
def extract_sift(images):

    sift = cv2.SIFT_create(nfeatures=500)

    all_desc, per_img = [], []

    for img in images:

        _, desc = sift.detectAndCompute(img,None)

        per_img.append(desc)

        if desc is not None:
            all_desc.append(desc)

    return np.vstack(all_desc), per_img


# ─────────────────────
# BUILD HISTOGRAMS
# ─────────────────────
def build_histograms(per_img, kmeans):

    hists = []

    for desc in per_img:

        if desc is None:

            hists.append(np.zeros(K))

        else:

            labels = kmeans.predict(desc)

            hist,_ = np.histogram(
                labels,
                bins=np.arange(K+1)
            )

            hists.append(hist.astype(float))

    return normalize(np.array(hists), norm="l2")


# ─────────────────────
# BUILD MODEL
# ─────────────────────
def build():

    imgs, paths = load_images()

    all_desc, per_img = extract_sift(imgs)

    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=1000,
        random_state=42
    ).fit(all_desc)

    histograms = build_histograms(per_img, kmeans)

    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(f"{SAVE_PATH}/model.pkl","wb") as f:

        pickle.dump({

            "kmeans":kmeans,
            "histograms":histograms,
            "paths":paths

        },f)

    print("Model saved")


# ─────────────────────
# SEARCH IMAGE
# ─────────────────────
def search(query):

    with open(f"{SAVE_PATH}/model.pkl","rb") as f:
        data = pickle.load(f)

    img = cv2.resize(
        cv2.imread(query,0),
        IMAGE_SIZE
    )

    sift = cv2.SIFT_create(nfeatures=500)

    _, desc = sift.detectAndCompute(img,None)

    labels = data["kmeans"].predict(desc)

    hist,_ = np.histogram(
        labels,
        bins=np.arange(K+1)
    )

    hist = normalize(
        hist.reshape(1,-1),
        norm="l2"
    )

    scores = cosine_similarity(
        hist,
        data["histograms"]
    )[0]

    ranked = np.argsort(scores)[::-1][:TOP_N]

    results = []

    for i in ranked:

        print(
            os.path.basename(data["paths"][i]),
            scores[i]
        )

        results.append((
            data["paths"][i],
            scores[i]
        ))

    show_results.display_results(
        query,
        results,
        top_n=TOP_N
    )


# ─────────────────────
# RUN
# ─────────────────────
if __name__=="__main__":

    import sys

    # build model
    if len(sys.argv)==2 and sys.argv[1]=="build":

        build()

    # search image
    elif len(sys.argv)==3 and sys.argv[1]=="search":

        search(sys.argv[2])

    # wrong command
    else:

        print(
            "python image_similarity_search.py build"
        )

        print(
            "python image_similarity_search.py search image.jpg"
        )