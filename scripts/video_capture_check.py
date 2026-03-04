import cv2, sys
p = r"W:\ADStudy\VF AD Blinded\Early Tongue Training\AD128.avi"
cap = cv2.VideoCapture(p)
print("opened:", cap.isOpened())
if cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    print("read:", ret, None if frame is None else frame.shape)
    cap.release()
