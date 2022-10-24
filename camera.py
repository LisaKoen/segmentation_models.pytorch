

import cv2


# def clearCapture(capture):
#     capture.release()
#     cv2.destroyAllWindows()
#
#
# def countCameras():
#     n = 0
#     for i in range(10):
#         try:
#             cap = cv2.VideoCapture(i)
#             ret, frame = cap.read()
#             cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             clearCapture(cap)
#             n += 1
#         except:
#             clearCapture(cap)
#             break
#     return n


if __name__ == '__main__':

    # print(countCameras())

    cap = cv2.VideoCapture(0)
    # check if we succeeded
    print("finished capture command")
    if (cap.isOpened()):
        print("Capture succeeded2")
        ret, frame = cap.read()
        print("ret '" + ret  + "'")
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()
    cv2.destroyAllWindows()
