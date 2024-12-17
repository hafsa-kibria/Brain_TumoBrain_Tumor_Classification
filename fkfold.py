import os, shutil
import PIL
from PIL import Image
import random


def MinDictCount(path):
    dict = {}
    img_count = []
    for idx, (root, dirs, files) in enumerate(os.walk(path, topdown=True)):
        if idx != 0:
            # folder name
            dict[root.split('\\')[-1]] = {}
            # image list
            dict[root.split('\\')[-1]]['image_list'] = os.listdir(root)
            # number of images in this folder
            dict[root.split('\\')[-1]]['image_count'] = len(os.listdir(root))
            img_count.append(len(os.listdir(root)))
    return dict, max(img_count)


def CV_Count(min, dict, trainRate=0.7, valRate=0.1):
    test_rate = 1 - trainRate - valRate
    for idx, items in enumerate(dict):
        dict[items]['Train'] = round(dict[items]['image_count'] * trainRate)
        dict[items]['Val'] = round(dict[items]['image_count'] * valRate)
        dict[items]['Test'] = round(dict[items]['image_count'] * test_rate)

    return dict


def CreateFold(fold_num, dict, source_path, dest_path):
    data_path = os.path.join(dest_path, 'Data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    folder_name = ['Train', 'Test', 'Val']
    for name in folder_name:
        folder_path = os.path.join(data_path, name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        for id in range(1, fold_num + 1):
            fold_path = os.path.join(folder_path, 'fold_' + str(id))
            if not os.path.exists(fold_path):
                os.mkdir(fold_path)


def CrossValidation(dict, path, dest_path, fold_num,
                    resize=(124, 124),
                    random_state=True,
                    video=False,
                    nifti=False,
                    Images=True,
                    image_format='.jpg',
                    audio=False,
                    audio_file_format='.wav'):
    for classes in dict:
        source_folder_path = os.path.join(path, classes)
        split = round(dict[classes]['image_count'] / fold_num)
        # print(split)
        print(classes)
        copy_image_list = dict[classes]['image_list'].copy()
        if random_state:
            copy_image_list = random.sample(copy_image_list, len(copy_image_list))
        image_list = copy_image_list + copy_image_list

        for folds in range(0, fold_num):
            for images in image_list[(0 + (folds * split)): (dict[classes]['Train'] + 1 + (folds * split))]:
                image_path = os.path.join(source_folder_path, images)
                dest_folder_path = os.path.join(dest_path, 'Data', 'Train', 'fold_' + str(folds + 1), classes)
                if not os.path.exists(dest_folder_path):
                    os.mkdir(dest_folder_path)
                if video == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + '.avi')
                    shutil.copy(image_path, dest_image_path)
                elif nifti == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + '.nii')
                    shutil.copy(image_path, dest_image_path)
                elif audio == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + audio_file_format)
                    shutil.copy(image_path, dest_image_path)
                elif Images == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + image_format)
                    image = Image.open(image_path)
                    try:
                        image = image.resize(resize, PIL.Image.Resampling.LANCZOS)
                        image = image.convert('RGB')
                        image.save(dest_image_path)
                    except OSError:
                        print(image_path)
                        continue

            for images in image_list[(dict[classes]['Train'] + 1 + (folds * split)): (
                    dict[classes]['Train'] + dict[classes]['Val'] + 1 + (folds * split))]:
                image_path = os.path.join(source_folder_path, images)
                dest_folder_path = os.path.join(dest_path, 'Data', 'Val', 'fold_' + str(folds + 1), classes)
                if not os.path.exists(dest_folder_path):
                    os.mkdir(dest_folder_path)

                if video == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + '.avi')
                    shutil.copy(image_path, dest_image_path)
                elif nifti == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + '.nii')
                    shutil.copy(image_path, dest_image_path)
                elif audio == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + audio_file_format)
                    shutil.copy(image_path, dest_image_path)


                elif Images == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + image_format)
                    image = Image.open(image_path)
                    try:
                        image = image.resize(resize, PIL.Image.Resampling.LANCZOS)
                        image = image.convert('RGB')
                        image.save(dest_image_path)
                    except OSError:
                        print(image_path)
                        continue

            for images in image_list[((dict[classes]['Train'] + dict[classes]['Val'] + 1 + (folds * split))): (
                    dict[classes]['Train'] + dict[classes]['Test'] + dict[classes]['Val'] + 1 + (folds * split))]:
                image_path = os.path.join(source_folder_path, images)
                dest_folder_path = os.path.join(dest_path, 'Data', 'Test', 'fold_' + str(folds + 1), classes)
                if not os.path.exists(dest_folder_path):
                    os.mkdir(dest_folder_path)

                if video == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + '.avi')
                    shutil.copy(image_path, dest_image_path)
                elif nifti == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + '.nii')
                    shutil.copy(image_path, dest_image_path)
                elif audio == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + audio_file_format)
                    shutil.copy(image_path, dest_image_path)
                elif Images == True:
                    dest_image_path = os.path.join(dest_folder_path, images.split('.')[0] + image_format)
                    image = Image.open(image_path)
                    try:
                        image = image.resize(resize, PIL.Image.Resampling.LANCZOS)
                        image = image.convert('RGB')
                        image.save(dest_image_path)
                    except OSError:
                        print(image_path)
                        continue


def main():
    ### Path of the Source folder
    path = r"D:\5data"

    ### Path of the destination folder
    dest_path = r"D:\kfold"

    # Creating a dictionary with folder name, image list, train, test,
    dict, max = MinDictCount(path)
    # print(max)

    # specify the trainRate, and valRate only
    dict = CV_Count(min, dict, trainRate=0.8, valRate=0.1)
    # print(round(max*0.7))

    # cross fold number is given 5 for now, you can use 5, 20, 7 as you want
    CreateFold(5, dict, path, dest_path)
    # for classes in dict:
    #     print(dict[classes]['Train'])

    # ross fold number is given 5 for now, random_state is true if you want to randomly sample total dataset and
    # then you want to seperate it for train test val
    CrossValidation(dict, path, dest_path, 5, resize=(124, 124),
                    random_state=False, video=False, nifti=False,
                    Images=True,
                    image_format='.jpg',
                    audio=False,
                    audio_file_format='.wav'
                    )

if __name__ == main():
    main()


