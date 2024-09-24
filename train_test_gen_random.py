import os
import random

def list_directory_contents(directory):
    contents = os.listdir(directory)
    filtered_contents = []
    for item in contents:
        if item == '.DS_Store':
            continue
        if item.endswith('-depth.png'):
            continue
        if item.endswith('-meta.json'):
            continue
        if item == 'camera.json':
            continue
        if item == 'meta.json':
            continue
        filtered_contents.append(item)
    return filtered_contents

def train_test_split(n, split):
    indices = list(range(n))
    random.shuffle(indices)
    num_test = int(split * n)

    test_indices = indices[:num_test]
    train_indices = indices[num_test:]
    
    return train_indices, test_indices

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

path = '/home/bham/Desktop/valve-6d-poseestimation/data_storage/Dataset/datasets/TNS_valve_dataset_kinect/data'
split_ratio = 0.2

folders = list_directory_contents(path)

all_train_components = []
all_test_components = []

for folder_name in folders:
    each_dir = os.path.join(path, folder_name)
    files = list_directory_contents(each_dir)

    train_list, test_list = train_test_split(len(files), split_ratio)
    
    train_list = sorted(train_list)
    test_list = sorted(test_list)

    for train_comp in train_list:
        all_train_components.append(str(folder_name)+'/'+str(train_comp))
    for test_comp in test_list:
        all_test_components.append(str(folder_name)+'/'+str(test_comp))

write_to_file('train.txt', all_train_components)
write_to_file('test.txt', all_test_components)