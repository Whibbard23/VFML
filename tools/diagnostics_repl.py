import cv2, numpy as np, os
import pathlib

p = pathlib.Path(r"C:\Users\Connor Lab\Desktop\VFML\data\crops\train\crop_test")
p.mkdir(parents=True, exist_ok=True)

img = (np.ones((100,100), dtype='uint8') * 127)

print("Attempting write...")
ok = cv2.imwrite(str(p/"test.png"), img)
print("Success:", ok)
