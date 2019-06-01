import os

root = "/projects/reverse-image-search/evaluation/confusion"

training_map = {
    'breaker_box': [],
    'sump_pump': [],
    'washer': []
}

import copy

validation_map = copy.deepcopy(training_map)

category_to_idx = {
    'breaker_box': 0,
    'sump_pump': 1,
    'washer': 2
}


def counts(data_dir, map):
    from os.path import isfile, join

    counter = 0
    for d in os.listdir(data_dir):
        if not isfile(join(data_dir, d)):
            for root, dirs, files in os.walk(join(data_dir, d)):
                for f in files:
                    map[d].append({
                        'idx': counter,
                        'label': category_to_idx[d],
                        'filename': root + d + f,
                        'filepath': root + '/' + f
                    })
                counter += 1


counts(root + '/data/household/train/', training_map)
counts(root + '/data/household/validation/', validation_map)

print("%table type\tclass\tcount")
for k in training_map:
    print("train\t" + k + "\t" + str(len(training_map[k])))

for k in validation_map:
    print("validation\t" + k + "\t" + str(len(validation_map[k])))

