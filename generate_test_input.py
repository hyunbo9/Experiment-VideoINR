import os 
import shutil

test_set = './adobe240/frame/test'
test_folder_list = os.listdir(test_set)

test_input_folder = './adobe240/frame/test_input'
os.makedirs(test_input_folder, exist_ok=True)


for test_folder in test_folder_list:
    test_folder_path = os.path.join(test_set, test_folder)

    test_files = os.listdir(test_folder_path)
    test_files_path = [os.path.join(test_folder_path, test_file) for test_file in test_files]


    os.makedirs(os.path.join(test_input_folder, test_folder), exist_ok=False)
    for test_file_path in test_files_path:
        data_num = int(os.path.basename(test_file_path).split('.')[0])
        if data_num % 8 == 0:
            target_fodler = os.path.join(test_input_folder, test_folder)
            shutil.copy(test_file_path, os.path.join(target_fodler, "img_{:04}.png".format(data_num//8)))


