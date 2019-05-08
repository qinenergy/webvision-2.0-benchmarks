import json

with open("top5_5k.json", "r") as f:
    x=json.load(f)

with open("label_5k.json", "r") as f:
    labels=json.load(f)

assert(len(x)==len(labels))

# Top1 score
summ = 0.
for i, pred in enumerate(x):
    if pred[0]!=labels[i]:
        summ += 1.

print("Top1:", summ/len(labels))

# Top5 score
summ = 0.
for i, pred in enumerate(x):
    if labels[i] not in pred:
        summ += 1.

print("Top5:", summ/len(labels))



# Top1 score
error = [0.] * 5000
base_error = [0.] * 5000

for i, pred in enumerate(x):
    if labels[i] != pred[0]:
        error[labels[i]]+=1.
    base_error[labels[i]]+=1.

sumsum = 0.
res = [0.] * 5000
for i in range(5000):
    sumsum += error[i]/base_error[i]
    res[i] = error[i]/base_error[i]

print("Balanced Top1:",sumsum/5000)


# Top5 score
error = [0.] * 5000
base_error = [0.] * 5000

for i, pred in enumerate(x):
    if labels[i] not in pred:
        error[labels[i]]+=1.
    base_error[labels[i]]+=1.

sumsum = 0.
for i in range(5000):
    sumsum += error[i]/base_error[i]

print("Balanced Top5:", sumsum/5000)








