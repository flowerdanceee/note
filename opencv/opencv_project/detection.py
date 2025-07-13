import cv2
import numpy as np


def filter_mask(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # Fill any small holes
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations=2)
    return dilation


def at_exit(point):
    try:
        if exit_mask[point[1]][point[0], 0] != 0:
            return True
    except:
        return True
    return False


def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return (cx, cy)


EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]
])
SHAPE = (720, 1280, 3)
base = np.zeros(SHAPE, dtype='uint8')
exit_mask = cv2.fillPoly(base, EXIT_PTS, (66, 183, 42))

cap = cv2.VideoCapture('road.mp4')
# 背景分割器对象
mog = cv2.createBackgroundSubtractorMOG2(500)

i = 0
while True:
    i = i + 1
    ret, frame = cap.read()
    fgmask = mog.apply(frame, 0.001)
    _, thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
    fgmask = filter_mask(thresh)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    list_1 = []
    for contour in contours:
        list_2 = []
        (x, y, w, h) = cv2.boundingRect(contour)
        # On the exit, we add some filtering by height, width and add centroid.
        point = get_centroid(x, y, w, h)
        contour_valid = (w >= 35) and (h >= 35) and (not at_exit(point))
        if contour_valid:
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (255, 192, 0), 1)
            cv2.circle(frame, point, 2, (0, 0, 255), -1)

    cv2.putText(frame, ("Frame: {total} ".format(total=i)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    cv2.putText(frame, ("Big vehicles passed:"), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    cv2.putText(frame, ("Small vehicles passed:"), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
    frame = cv2.addWeighted(exit_mask, 1, frame, 1, 0, frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(30) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
