# Example code to upsample the training data list
# You can insert it to get_data method in class Imagenet5k

from collections import defaultdict
import numpy as np

 
imglist = [("file1", 0), ("file2", 0), ("file3", 1), ("file4", 2), ("file5", 2), ("file6", 2)]


idxs_bygroup = defaultdict(list)
for i, (_, label) in enumerate(imglist):
    idxs_bygroup[label].append(i)


keys = list(idxs_bygroup.keys())
maxlength = np.max([len(idxs_bygroup[k]) for k in keys])
updated_idxs = []


for k in keys:
    length = len(idxs_bygroup[k])
    if length == maxlength:
        updated_idxs.extend(idxs_bygroup[k])
    else:
        grouplist = np.random.choice(idxs_bygroup[k], maxlength)
        updated_idxs.extend(grouplist)


np.random.shuffle(updated_idxs)

print("Before\n", np.array(imglist))
print("After\n", np.array(imglist)[updated_idxs])
