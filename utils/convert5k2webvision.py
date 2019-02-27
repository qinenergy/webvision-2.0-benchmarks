import json

with open("map.json", "r") as f:
    mapping = json.load(f)

with open("../top5.json", "r") as f:
    preds = json.load(f)


for i, batch in enumerate(preds):
    for j, num in enumerate(batch):
        preds[i][j] = mapping[num]


with open("top5_resnet2x.json", "w") as f:
    json.dump(preds, f)
    
