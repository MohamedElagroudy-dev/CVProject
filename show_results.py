"""
show_results.py
---------------
Visualization module for image similarity search results
"""

import os
import cv2
import numpy as np

# ── config ──────────────────────────────
TOP_N = 5  # how many results to show
# ────────────────────────────────────────


def load_color_image(path, size=(256, 256)):
    """Load and resize a color image."""
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(img, size)


def add_label(img, text, score, is_query=False):
    """Draw a label bar at the bottom of an image tile."""
    h, w = img.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)

    if is_query:
        bar[:] = (60, 60, 60)
        label = "QUERY IMAGE"
        color = (255, 255, 255)
    else:
        bar[:] = (30, 30, 30)
        label = f"{os.path.basename(text)}  {score:.3f}"
        color = (80, 220, 120)

    cv2.putText(bar, label, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def build_display(query_path, results, top_n=TOP_N):
    """
    Build visualization canvas with query image and search results.
    
    Layout:
      Row 1 (top):   query image (centered, larger)
      Row 2 (bottom): top-5 results side by side
    
    Args:
        query_path: path to query image
        results: list of tuples (image_path, score)
        top_n: number of results to display
    
    Returns:
        canvas: numpy array of the combined display
    """
    TILE = (256, 256)   # width, height of each result tile
    QUERY = (384, 384)  # query image shown bigger

    # ── query tile ──
    q_img = load_color_image(query_path, QUERY)
    q_img = add_label(q_img, query_path, 0, is_query=True)

    # ── result tiles ──
    result_tiles = []
    for path, score in results[:top_n]:
        tile = load_color_image(path, TILE)
        tile = add_label(tile, path, score)
        result_tiles.append(tile)

    # pad result row to top_n tiles
    while len(result_tiles) < top_n:
        blank = np.zeros((TILE[1] + 40, TILE[0], 3), dtype=np.uint8)
        result_tiles.append(blank)

    result_row = np.hstack(result_tiles)   # shape: (296, 1280, 3)
    canvas_w = result_row.shape[1]

    # center the query image on the canvas width
    q_h, q_w = q_img.shape[:2]
    pad_left = (canvas_w - q_w) // 2
    pad_right = canvas_w - q_w - pad_left
    query_row = np.hstack([
        np.zeros((q_h, pad_left, 3), dtype=np.uint8),
        q_img,
        np.zeros((q_h, pad_right, 3), dtype=np.uint8),
    ])

    # divider line
    divider = np.full((6, canvas_w, 3), 50, dtype=np.uint8)

    # header bar
    header = np.zeros((36, canvas_w, 3), dtype=np.uint8)
    cv2.putText(header, "Image Similarity Search Results",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 1, cv2.LINE_AA)

    canvas = np.vstack([header, query_row, divider, result_row])
    return canvas


def display_results(query_path, results, top_n=TOP_N, save_output=True):
    """
    Display results in a window and optionally save to file.
    
    Args:
        query_path: path to query image
        results: list of tuples (image_path, score)
        top_n: number of results to display
        save_output: whether to save the result image
    
    Returns:
        canvas: the generated visualization
    """
    canvas = build_display(query_path, results, top_n)
    
    print("\nShowing results window — press any key to close it.")
    cv2.imshow("Image Similarity Search", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if save_output:
        out_file = "search_results.jpg"
        cv2.imwrite(out_file, canvas)
        print(f"Result also saved as '{out_file}'")
    
    return canvas