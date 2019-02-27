import json
with open("top5_resnet2x.json", "r") as f:
    res = json.load(f)

with open("/raid/qwang/webvision2018_test/meta/val.txt", "r") as f:
    names = [x.split()[0] for x in f.readlines()]

with open("predictions.txt", "w") as f:
    for name, r in zip(names, res):
        f.write(name+" "+str(r[0])+"    "+str(r[1])+"    "+str(r[2])+"    "+str(r[3])+"    "+str(r[4])+"\n"
) 
