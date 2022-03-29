import random
import json

num_samples = 1000
num_joints = 6
data = []

for i in range(num_samples):
    q = [random.random() for j in range(num_joints)]
    qd = [random.random() for j in range(num_joints)]
    qdd = [random.random() for j in range(num_joints)]
    q_aux_d = [random.random() for j in range(num_joints)]

    d = {'q':q, 'qd':qd, 'qdd':qdd,'q_aux_d':q_aux_d}
    data.append(d)

with open('sample_input.json', 'w') as f:
    json.dump(data, f)

