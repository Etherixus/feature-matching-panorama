# Feature Matching, Object Localization, and Panorama Stitching

Two OpenCV pipelines built on local feature detection: one stitches
overlapping photographs into a panorama, the other locates a cropped face
inside a full scene under rotation and perspective change.

Built for CSCI 431 (Introduction to Computer Vision), Rochester Institute of
Technology, Spring 2026.

## Task 1 — Panorama stitching (`task_1.py`)

Takes a reference and a query image and warps the second onto the first.

1. **Detect and describe** with a selectable backend — SIFT and SURF use L2
   distance, ORB uses Hamming, and `create_detector()` returns the matching
   norm alongside the detector so the rest of the pipeline stays agnostic.
2. **Match** descriptors with a brute-force matcher, `knnMatch` with k=2.
3. **Filter** with Lowe's ratio test, discarding matches whose best and
   second-best distances are too close to separate.
4. **Estimate homography** with `findHomography` under RANSAC, 5.0 px
   reprojection threshold.
5. **Warp and compose** — project the query image's corners through H to find
   the output canvas bounds, build a translation matrix T so nothing lands at
   negative coordinates, then `warpPerspective` with `T @ H`.

Run against two scenes, a gym and a parking lot, with SIFT and ORB each.
Results are in `outputs/`.

## Task 2 — Object localization (`task_2.py`)

Finds a cropped face within a larger image.

- **Keypoint visualization** — ORB with 1000 features, drawn both plainly and
  with `DRAW_RICH_KEYPOINTS` to show scale and orientation.
- **Rotation invariance** — matches the face against a rotated copy of itself
  using `BFMatcher(NORM_HAMMING, crossCheck=True)`, confirming ORB's
  orientation compensation holds.
- **Localization** — raises the budget to 10,000 features for the full scene,
  matches with KNN plus ratio test, fits a homography under RANSAC, then
  projects the face's corner points through it with `perspectiveTransform`
  to draw the quadrilateral where the face sits in the scene.

## Layout

```
task_1.py                 Panorama stitching
task_2.py                 Keypoints, rotation matching, localization
reference_*.jpg           Panorama inputs
query_*.jpg
face.jpg, full_image.jpg  Task 2 inputs
rotated_img.jpg
outputs/                  Generated results for both tasks
```

## Running it

```bash
pip install opencv-python numpy matplotlib
python task_1.py
python task_2.py
```

Both scripts read their images from the working directory, so run them from
the repository root. Note that `create_detector("SURF")` needs
`opencv-contrib-python` built with non-free algorithms enabled; SIFT and ORB
work with the standard wheel, and those are what the committed outputs use.
