import cv2
import numpy as np


# --------------------------------------------------
# Create feature detector based on selected method
# Also return the correct distance metric for matching
# --------------------------------------------------
def create_detector(method):
    method = method.upper()

    # SIFT uses L2 (Euclidean) distance
    if method == "SIFT":
        return cv2.SIFT_create(), cv2.NORM_L2

    # ORB uses Hamming distance (binary descriptors)
    elif method == "ORB":
        return cv2.ORB_create(nfeatures=3000), cv2.NORM_HAMMING

    # SURF also uses L2 distance
    elif method == "SURF":
        return cv2.xfeatures2d.SURF_create(), cv2.NORM_L2

    else:
        raise ValueError("Method must be SIFT, SURF, or ORB")


# --------------------------------------------------
# Main stitching function
# Takes reference image + query image
# Aligns query image to reference using feature matching
# --------------------------------------------------
def stitch_images(ref_img, query_img, method="SIFT"):

    # Convert both images to grayscale (required for feature detection)
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_query = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

    # Create feature detector + appropriate distance metric
    detector, norm_type = create_detector(method)

    # Detect keypoints and compute descriptors
    # kp = keypoints (locations), des = feature vectors
    kp1, des1 = detector.detectAndCompute(gray_ref, None)
    kp2, des2 = detector.detectAndCompute(gray_query, None)

    # If descriptors cannot be computed, stop early
    if des1 is None or des2 is None:
        print(f"{method}: descriptors could not be computed.")
        return None

    # --------------------------------------------------
    # Feature Matching (Brute Force)
    # --------------------------------------------------
    # knnMatch finds the 2 best matches for each descriptor
    bf = cv2.BFMatcher(norm_type)
    matches = bf.knnMatch(des2, des1, k=2)  
    # NOTE: query → reference matching

    # --------------------------------------------------
    # Lowe's Ratio Test
    # Filters out weak/ambiguous matches
    # --------------------------------------------------
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    print(f"{method}: good matches = {len(good)}")

    # Need at least 4 matches to compute homography
    if len(good) < 4:
        print(f"{method}: not enough good matches.")
        return None

    # --------------------------------------------------
    # Extract matched keypoint coordinates
    # --------------------------------------------------
    # src_pts = points from query image
    # dst_pts = corresponding points in reference image
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # --------------------------------------------------
    # Compute Homography using RANSAC
    # This finds the transformation between the images
    # while rejecting outliers
    # --------------------------------------------------
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print(f"{method}: homography failed.")
        return None

    # --------------------------------------------------
    # Determine size of output panorama
    # --------------------------------------------------
    h1, w1 = ref_img.shape[:2]
    h2, w2 = query_img.shape[:2]

    # Define corner points of both images
    corners_query = np.float32([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ]).reshape(-1, 1, 2)

    corners_ref = np.float32([
        [0, 0],
        [0, h1],
        [w1, h1],
        [w1, 0]
    ]).reshape(-1, 1, 2)

    # Transform query image corners using homography
    warped_corners_query = cv2.perspectiveTransform(corners_query, H)

    # Combine all corners to compute full bounding box
    all_corners = np.concatenate((corners_ref, warped_corners_query), axis=0)

    # Find min/max coordinates
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # --------------------------------------------------
    # Translation matrix (to avoid negative coordinates)
    # --------------------------------------------------
    translation = [-xmin, -ymin]
    T = np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])

    # --------------------------------------------------
    # Warp query image into panorama space
    # --------------------------------------------------
    panorama = cv2.warpPerspective(query_img, T @ H, (xmax - xmin, ymax - ymin))

    # Place reference image into the panorama
    panorama[
        translation[1]:h1 + translation[1],
        translation[0]:w1 + translation[0]
    ] = ref_img

    return panorama


# --------------------------------------------------
# Run stitching for both images (gym / parking)
# --------------------------------------------------
def run_case(ref_path, query_path, prefix):

    # Load images
    ref_img = cv2.imread(ref_path)
    query_img = cv2.imread(query_path)

    if ref_img is None or query_img is None:
        print(f"Could not load images for {prefix}")
        return

    # Methods required by assignment
    methods = ["SIFT", "ORB"]

    for method in methods:
        try:
            # Perform stitching
            result = stitch_images(ref_img, query_img, method)

            if result is not None:
                # Save output image
                out_name = f"{prefix}_{method}_output.jpg"
                cv2.imwrite(out_name, result)

                # Display result
                cv2.imshow(f"{prefix} - {method}", result)

                print(f"Saved {out_name}")

        except Exception as e:
            print(f"{method} failed: {e}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------
# Main execution
# --------------------------------------------------
if __name__ == "__main__":

    # Run stitching on both datasets
    run_case("reference_gym.jpg", "query_gym.jpg", "gym")
    run_case("reference_parking.jpg", "query_parking.jpg", "parking")