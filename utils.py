import numpy as np
import cv2

def mask_to_polygons(mask):
    """
    Converts a binary mask to a list of normalized polygons represented as strings.

    Parameters:
        mask (np.ndarray): A binary mask represented as a 2D NumPy array of
            shape `(H, W)`, where H and W are the height and width of
            the mask, respectively.

    Returns:
        List[str]: A list of polygons, where each polygon is represented by a
            string containing the normalized `x`, `y` coordinates of the points.
            Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
            are excluded from the output.
    """
    height, width = mask.shape
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:
            normalized_contour = contour.astype(np.float32) / [width, height]
            flattened_contour = normalized_contour.reshape(-1)
            polygon_str = ' '.join(map(str, flattened_contour))
            polygons.append(polygon_str)
    
    return polygons

def polygon_to_mask(polygon, resolution_wh):
    """Generate a mask from a polygon.

    Args:
        polygon (np.ndarray): The polygon for which the mask should be generated,
            given as a list of vertices.
        resolution_wh (Tuple[int, int]): The width and height of the desired resolution.

    Returns:
        np.ndarray: The generated 2D mask, where the polygon is marked with
            `1`'s and the rest is filled with `0`'s.
    """
    width, height = resolution_wh
    mask = np.zeros((height, width))

    cv2.fillPoly(mask, [polygon], color=1)
    return mask