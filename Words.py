import cv2
import numpy as np

def Sorting_Key(rect):
    global Lines, Size

    x, y, w, h = rect

    cx = x + int(w / 2)
    cy = y + int(h / 2)

    for i, (upper, lower) in enumerate(Lines):
        if not any([all([upper > y + h, lower > y + h]), all([upper < y, lower < y])]):
            return cx + ((i + 1) * Size)

def Split(Image):
    global Lines, Size

    # Convert the input image to grayscale
    gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

    # Apply morphological operations to enhance the text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Ensure that there are no zero values in the morph image
    for i in range(morph.shape[0]):
        for j in range(morph.shape[1]):
            if not morph[i][j]:
                morph[i][j] = 1

    # Divide the grayscale image by the morphological result
    div = gray / morph

    # Normalize the result to the range [0, 255]
    gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

    # Apply thresholding to obtain a binary image
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological dilation to further enhance the text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Remove small contours (noise) based on area threshold
    i = 0
    contours = list(contours)
    Length = len(contours)
    while i < Length:
        x, y, w, h = cv2.boundingRect(contours[i])
        if w * h <= 100:
            del contours[i]
            i -= 1
            Length -= 1
        i += 1

    # Compute horizontal projection to segment lines of text
    h_proj = np.sum(thresh, axis=1)

    upper = None
    lower = None
    Lines = []
    for i in range(h_proj.shape[0]):
        proj = h_proj[i]

        if proj != 0 and upper is None:
            upper = i
        elif proj == 0 and upper is not None and lower is None:
            lower = i
            if lower - upper >= 30:
                Lines.append([upper, lower])
            upper = None
            lower = None

    # Handle the case when the last line doesn't have a closing boundary
    if upper:
        Lines.append([upper, h_proj.shape[0] - 1])

    Size = thresh.shape[1]

    # Extract bounding rectangles of text regions
    bounding_rects = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the bounding rectangle is within a text line region
        for upper, lower in Lines:
            if not any([all([upper > y + h, lower > y + h]), all([upper < y, lower < y])]):
                bounding_rects.append([x, y, w, h])

    # Combine overlapping bounding rectangles
    i = 0
    Length = len(bounding_rects)
    while i < Length:
        x, y, w, h = bounding_rects[i]
        j = 0

        while j < Length:
            distancex = abs(bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2]))
            distancey = abs(bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3]))

            threshx = max(abs(bounding_rects[j][0] - (bounding_rects[i][0] + bounding_rects[i][2])),
                          abs(bounding_rects[j][0] - bounding_rects[i][0]),
                          abs((bounding_rects[j][0] + bounding_rects[j][2]) - bounding_rects[i][0]),
                          abs((bounding_rects[j][0] + bounding_rects[j][2]) - (bounding_rects[i][0] + bounding_rects[i][2])))

            threshy = max(abs(bounding_rects[j][1] - (bounding_rects[i][1] + bounding_rects[i][3])),
                          abs(bounding_rects[j][1] - bounding_rects[i][1]),
                          abs((bounding_rects[j][1] + bounding_rects[j][3]) - bounding_rects[i][1]),
                          abs((bounding_rects[j][1] + bounding_rects[j][3]) - (bounding_rects[i][1] + bounding_rects[i][3])))

            # Check for overlap conditions and combine rectangles
            if i != j and any([all([not any([all([bounding_rects[j][1] > y + h, bounding_rects[j][1] + bounding_rects[j][3] > y + h]), all([bounding_rects[j][1] < y, bounding_rects[j][1] + bounding_rects[j][3] < y])]),
                                   not any([all([bounding_rects[j][0] > x + w, bounding_rects[j][0] + bounding_rects[j][2] > x + w]), all([bounding_rects[j][0] < x, bounding_rects[j][0] + bounding_rects[j][2] < x])])]),
                              all([distancex <= 10, bounding_rects[i][3] + bounding_rects[j][3] + 10 >= threshy]), all([bounding_rects[i][2] + bounding_rects[j][2] + 10 >= threshx, distancey <= 10])]):

                x = min(bounding_rects[i][0], bounding_rects[j][0])
                w = max(bounding_rects[i][0] + bounding_rects[i][2], bounding_rects[j][0] + bounding_rects[j][2]) - x
                y = min(bounding_rects[i][1], bounding_rects[j][1])
                h = max(bounding_rects[i][1] + bounding_rects[i][3], bounding_rects[j][1] + bounding_rects[j][3]) - y

                bounding_rects[i] = [x, y, w, h]
                del bounding_rects[j]
                i = -1
                Length -= 1
                break

            j += 1
        i += 1

    # Sort bounding rectangles based on their x-coordinate using a custom sorting key
    bounding_rects.sort(key=Sorting_Key)

    Words = []
    # Extract individual words from the bounding rectangles
    for x, y, w, h in bounding_rects:
        crop = Image[y:y + h, x:x + w]

        # Convert the cropped region to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Apply morphological operations to enhance the text in the cropped region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # Ensure that there are no zero values in the morph image
        for i in range(morph.shape[0]):
            for j in range(morph.shape[1]):
                if not morph[i][j]:
                    morph[i][j] = 1

        # Divide the grayscale image by the morphological result
        div = gray / morph
        gray = np.array(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX), np.uint8)

        # Apply thresholding to obtain a binary image
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the binary image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = np.vstack(contours)

        # Find the minimum area rectangle enclosing the contours
        rect = cv2.minAreaRect(contours)
        Box = cv2.boxPoints(rect)

        # Reorder the points of the rectangle to ensure correct perspective transformation
        index = np.argmin(np.sum(Box, axis=1))
        box = []
        box.extend(Box[index:])
        box.extend(Box[0:index])
        box = np.int0(box)

        # Determine the shape of the resulting perspective-transformed image
        shape = (box[1][0] - box[0][0], box[3][1] - box[0][1])

        # Define source and destination points for perspective transformation
        src = np.float32(box)
        dst = np.array([[0, 0], [shape[0], 0], [shape[0], shape[1]], [0, shape[1]]], np.float32)

        # Perform perspective transformation and invert the result
        M = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.bitwise_not(cv2.warpPerspective(cv2.bitwise_not(crop), M, shape))

        # Append the resulting word image to the list
        Words.append(warp.copy())

    return Words
