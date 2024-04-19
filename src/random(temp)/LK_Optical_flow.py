import sys
import numpy as np
import cv2 as cv


def LK(cap, old_frame, frame, old_gray, frame_gray, p0, lk_params):
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points, st is the status of tracking, 1 is tracking successful
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # sys.exit()

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(
            mask, (int(a), int(b)), (int(c), int(d)), cv.color[i].tolist(), 2
        )
        frame = cv.circle(frame, (int(a), int(b)), 5, cv.color[i].tolist(), -1)
    img = cv.add(frame, mask)

    cv.imshow("frame", img)
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    cv.destroyAllWindows()


def Shi_Tomasi(cap):
    # cv.samples.findFile("vtest.avi")
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    while 1:
        ret, frame2 = cap.read()
        if not ret:
            print("No frames grabbed!")
            break

        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(
            prvs,
            next,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=4,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        np.set_printoptions(threshold=np.inf)
        print(flow)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        # cv.imshow("frame2", bgr)
        k = cv.waitKey(30) & 0xFF
        if k == 27:
            break
        elif k == ord("s"):
            cv.imwrite("opticalfb.png", frame2)
            cv.imwrite("opticalhsv.png", bgr)
        prvs = next

    cv.destroyAllWindows()
