from pathlib import Path
from typing import Any, Dict, List, Tuple
import shutil

import cv2
import numpy as np
from numpy import typing as npt
from scipy.signal import savgol_filter
from tqdm import tqdm

def copy_unlabeled_images(images_raw_path: Path, labels_path: Path, images_draw_path: Path, trajectory: List[Tuple[int, int]]) -> None:
    """
    Copies images from images_raw_path to images_draw_path if they do not have a corresponding label in labels_path.

    Args:
        images_raw_path: Path to the directory containing raw images.
        labels_path: Path to the directory containing label files.
        images_draw_path: Path to the directory where unlabeled images will be copied.
    """
    # Ensure output directory exists
    images_draw_path.mkdir(parents=True, exist_ok=True)

    # Get list of image filenames without extensions
    image_files = {f.stem for f in images_raw_path.glob("*.jpg")}
    label_files = {f.stem for f in labels_path.glob("*.txt")}  # Assuming labels are .txt files

    # Find images without corresponding labels
    unlabeled_images = image_files - label_files

    # Copy unlabeled images to images_draw_path
    for image_name in unlabeled_images:
        image = cv2.imread(images_raw_path / f"{image_name}.jpg", 1)

        for point in trajectory:
            draw_circle(image, point[0], point[1], 5, 5)

        cv2.imwrite(str(images_draw_path / f"{image_name}.jpg"), image)

        #src_image_path = images_raw_path / f"{image_name}.jpg"
        #dest_image_path = images_draw_path / f"{image_name}.jpg"
        #shutil.copy2(src_image_path, dest_image_path)


def smooth_trajectory(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    split_trajectory = [[points[0]]]
    current_split = 0
    for i in range(1, len(points)):
        distance = np.linalg.norm(np.array(points[i - 1]) - np.array(points[i]))
        if distance < 15:
            split_trajectory[current_split].append(points[i])
        else:
            split_trajectory.append([points[i]])
            current_split += 1

    new_X, new_Y = [], []

    for split in split_trajectory:
        X = np.array([p[0] for p in split])
        Y = np.array([p[1] for p in split])

        if len(Y) > 20:
            y_hat = savgol_filter(Y, 15, 3)
            Y = [int(round(y)) for y in y_hat]

        new_X.extend(X)
        new_Y.extend(Y)

    return list(zip(new_X, new_Y))


def get_pixel_coords(
    obj: Dict[str, Any], img_height: int, img_width: int
) -> Tuple[int, int, int, int]:
    coords = obj["relative_coordinates"]
    center_x = int(img_width * coords["center_x"])
    center_y = int(img_height * coords["center_y"])
    obj_width = int(img_width * coords["width"])
    obj_height = int(img_height * coords["height"])

    return center_x, center_y, obj_width, obj_height


def draw_circle(image: npt.NDArray[np.uint8], x: int, y: int, w: int, h: int) -> None:
    r = max(int(w / 2), int(h / 2))
    cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    cv2.circle(image, (x, y), 2, (0, 255, 0), -2)


def draw_box(image: npt.NDArray[np.uint8], x: int, y: int, w: int, h: int) -> None:
    cv2.rectangle(
        image,
        (int(x - w / 2), int(y + h / 2)),
        (int(x + w / 2), int(y - h / 2)),
        (255, 0, 255),
        2,
    )


def draw_line_fragment(
    image: npt.NDArray[np.uint8],
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (135, 0, 190),
    thickness: int = 2,
    max_distance: int = 30,
) -> npt.NDArray[np.uint8]:
    for i in range(1, len(points)):
        distance = np.linalg.norm(np.array(points[i - 1]) - np.array(points[i]))
        if distance < max_distance:
            image = cv2.line(image, points[i - 1], points[i], color, thickness)

    return image


def draw_glowing_line(
    img: npt.NDArray[np.uint8], points: List[Tuple[int, int]], max_distance: int
) -> npt.NDArray[np.uint8]:
    line_img = np.zeros_like(img)
    line_img = draw_line_fragment(
        line_img, points, color=(135, 0, 190), thickness=2, max_distance=max_distance
    )
    line_img = draw_line_fragment(
        line_img, points, color=(0, 0, 255), thickness=1, max_distance=max_distance
    )

    blurred_line = cv2.GaussianBlur(line_img, (5, 5), 0)

    img_gray = cv2.cvtColor(blurred_line, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)
    mask = np.stack([mask, mask, mask], axis=2)
    mask = np.where(mask > 0, 0.6, 1)
    img = img * mask

    return cv2.addWeighted(img.astype("uint8"), 1, blurred_line, 1.6, 0)


def load_labels(path: Path) -> List[Dict[str, Any]]:
    with path.open() as label_file:
        lines = label_file.readlines()

    objects = []
    for line in lines:
        class_id, center_x, center_y, width, height, conf = line.split()
        objects.append(
            {
                "class_id": int(class_id),
                "relative_coordinates": {
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "width": float(width),
                    "height": float(height),
                },
                "confidence": float(conf),
            }
        )

    return objects


def process_image(
    label_filename: Path,
    points: List[Tuple[int, int]],
    images_path: Path,
    ball_conf: float,
    max_distance: int,
) -> Tuple[npt.NDArray[np.uint8], List[Tuple[int, int]]]:
    image_filename = str(images_path / label_filename.with_suffix(".jpg").name)
    objects = load_labels(label_filename)

    image = cv2.imread(image_filename, 1)
    height, width, _ = image.shape

    obj_to_draw = None
    max_conf = 0
    for obj in objects:
        new_point = None
        if (
            obj["class_id"] == 0
            and obj["confidence"] >= ball_conf
            and obj["confidence"] > max_conf
        ):
            obj_to_draw = obj
            max_conf = obj["confidence"]

    if obj_to_draw:
        ball_x, ball_y, obj_width, obj_height = get_pixel_coords(
            obj_to_draw, height, width
        )
        new_point = ball_x, ball_y
        points.append(new_point)
        draw_circle(image, ball_x, ball_y, obj_width, obj_height)

    points = list(filter(lambda pt: pt is not None, points))
    #if len(points) > 30:
    #    points = smooth_trajectory(points)
#
    #image = draw_glowing_line(image, points, max_distance)

    for point in points:
        draw_circle(image, point[0], point[1], 5, 5)
    return image, points


def draw_trajectory(
    labels_dir: Path,
    images_path: Path,
    output_path: Path,
    ball_conf: float = 0.3,
    max_distance: int = 30,
) -> List[Tuple[int, int]]:
    label_files = {f.stem: f for f in labels_dir.glob("*.txt")}
    image_files = sorted(images_path.glob("*.jpg"), key=lambda x: int(x.stem))

    points = []
    progress_bar = tqdm(image_files)
    for img_id, image_file in enumerate(progress_bar):
        progress_bar.set_description("Drawing trajectories")

        label_file = label_files.get(image_file.stem, None)

        if label_file:
            image, points = process_image(
                label_file,
                points,
                images_path,
                ball_conf,
                max_distance,
            )
        else:
            image = cv2.imread(str(image_file), 1)
            for point in points:
                draw_circle(image, point[0], point[1], 5, 5)
        #cv2.imwrite(str(output_path / f"{img_id:05d}.jpg"), image)
        cv2.imwrite(str(output_path / f"{image_file.stem}.jpg"), image)

    # Copy image that doesn't reconize to image_draw folder
    #copy_unlabeled_images(images_path, labels_dir, output_path, points)

    return points
