import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Helper function: load image in RGB and grayscale
# --------------------------------------------------
def load_color_and_gray(path):
    img_bgr = cv2.imread(path)

    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return img_rgb, img_gray


# --------------------------------------------------
# Part A and B:
# Detect ORB keypoints and draw them
# --------------------------------------------------
def detect_and_draw_keypoints(face_rgb, face_gray):
    # ORB detector with 1000 features as requested
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2)

    keypoints, descriptors = orb.detectAndCompute(face_gray, None)

    # Draw normal keypoints
    img_keypoints = cv2.drawKeypoints(
        face_rgb,
        keypoints,
        None,
        flags=0
    )

    # Draw rich keypoints
    img_rich_keypoints = cv2.drawKeypoints(
        face_rgb,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return keypoints, descriptors, img_keypoints, img_rich_keypoints


# --------------------------------------------------
# Part C:
# Match face with rotated face using ORB + BFMatcher
# --------------------------------------------------
def match_rotated_face(face_rgb, face_gray, rotated_rgb, rotated_gray):
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2)

    kp1, des1 = orb.detectAndCompute(face_gray, None)
    kp2, des2 = orb.detectAndCompute(rotated_gray, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not compute descriptors for rotated face matching.")

    # BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Top 300 matches as requested
    best_matches = matches[:300]

    matched_image = cv2.drawMatches(
        face_rgb,
        kp1,
        rotated_rgb,
        kp2,
        best_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return matched_image, kp1, kp2, best_matches


# --------------------------------------------------
# Part D:
# Match face inside full image using
# ORB + KNN + ratio test + homography + RANSAC
# --------------------------------------------------
def match_face_in_full_image(face_rgb, face_gray, full_rgb, full_gray):
    # More features for the large full image
    orb = cv2.ORB_create(nfeatures=10000, scaleFactor=1.2)

    kp1, des1 = orb.detectAndCompute(face_gray, None)
    kp2, des2 = orb.detectAndCompute(full_gray, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not compute descriptors for full image matching.")

    # KNN matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print("Good matches before RANSAC:", len(good_matches))

    if len(good_matches) < 4:
        raise ValueError("Not enough good matches to compute homography.")

    # Face -> Full image point mapping
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography computation failed.")

    matches_mask = mask.ravel().tolist()

    # Keep only inlier matches
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
    print("Inlier matches after RANSAC:", len(inlier_matches))

    # Draw only inlier matches
    matches_image = cv2.drawMatches(
        face_rgb,
        kp1,
        full_rgb,
        kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Draw bounding box of detected face in full image
    h, w = face_gray.shape

    face_corners = np.float32([
        [0, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w - 1, 0]
    ]).reshape(-1, 1, 2)

    projected_corners = cv2.perspectiveTransform(face_corners, H)
    projected_corners = np.int32(projected_corners)

    full_with_box = full_rgb.copy()
    cv2.polylines(
        full_with_box,
        [projected_corners],
        True,
        (0, 255, 0),
        3
    )

    return matches_image, full_with_box, inlier_matches


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # Load images
    face_rgb, face_gray = load_color_and_gray("face.jpg")
    rotated_rgb, rotated_gray = load_color_and_gray("rotated_img.jpg")
    full_rgb, full_gray = load_color_and_gray("full_image.jpg")

    # -------------------------------
    # Part A and B
    # -------------------------------
    keypoints, descriptors, img_keypoints, img_rich_keypoints = detect_and_draw_keypoints(
        face_rgb, face_gray
    )

    print("Number of keypoints detected on face image:", len(keypoints))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_keypoints)
    plt.title("ORB Keypoints")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_rich_keypoints)
    plt.title("ORB Keypoints with Rich Details")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("task2_keypoints_output.jpg")
    plt.show()

    # -------------------------------
    # Part C
    # -------------------------------
    rotated_match_img, kp1, kp2, best_matches = match_rotated_face(
        face_rgb, face_gray, rotated_rgb, rotated_gray
    )

    print("Rotated face matches:", len(best_matches))

    plt.figure(figsize=(16, 8))
    plt.imshow(rotated_match_img)
    plt.title("Face vs Rotated Face - ORB Feature Matching")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("task2_rotated_face_matching.jpg")
    plt.show()

    # -------------------------------
    # Part D
    # -------------------------------
    full_match_img, full_with_box, inlier_matches = match_face_in_full_image(
        face_rgb, face_gray, full_rgb, full_gray
    )

    plt.figure(figsize=(18, 8))
    plt.imshow(full_match_img)
    plt.title("Face vs Full Image - Inlier Matches Only")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("task2_full_image_matching.jpg")
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(full_with_box)
    plt.title("Detected Face Region in Full Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("task2_face_detected_in_full_image.jpg")
    plt.show()


if __name__ == "__main__":
    main()