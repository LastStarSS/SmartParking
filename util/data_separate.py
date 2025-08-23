import os
import shutil

src = "D:/Donee/Coding/AICourse/YOLO_OBB/datasets/valid"
dest = "D:/Donee/Coding/AICourse/YOLO_OBB/datasets/data_val"

all_data = os.listdir(src)

for idx, filename in enumerate(all_data):
    if idx % 10 == 0:
        data_src = os.path.join(src, filename)
        data_dest = os.path.join(dest, filename)
        shutil.copyfile(data_src, data_dest)

print("Done")