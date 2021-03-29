import numpy as np

IDX_TO_CLASS = []
with open("./tiny-imagenet-200/wnids.txt") as f:
    for line in f:
        IDX_TO_CLASS.append(line.strip())
f.close()

CLASS_TO_LABEL = {}
with open("./tiny-imagenet-200/words.txt") as f:
    for line in f:
        (key, val) = line.split('\t')
        val = val.strip()
        CLASS_TO_LABEL[key] = val
f.close()

def indexToClass(idx):
    return IDX_TO_CLASS[idx]

def classToLabel(class_name):
    return CLASS_TO_LABEL[class_name]

def indexToLabel(idx):
    class_name = indexToClass(idx)
    return classToLabel(class_name)

def num_classes():
    return len(IDX_TO_CLASS)