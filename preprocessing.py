import os
import cv2
import numpy as np

data_path =r'Data\Test\fold_1'

categories = os.listdir(data_path)
print("Number of classes", categories)
noofClasses = len(categories)
print("Total number of classes", noofClasses)
print("Importing images")

labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories, labels)) #empty dictionary

print(label_dict)
print(categories)
print(labels)

img_size = 124
data = []
target = []

def clahe_function(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab_img)
    clahe_img = clahe.apply(lab_planes[0])
    lab_planes[0] = clahe_img
    updated_lab_img2 = cv2.merge(lab_planes)
    # Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    return CLAHE_img



for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Coverting the image into gray scale

            img = clahe_function(img)
            # img = ben_graham(img)
            imga = cv2.resize(img, (img_size, img_size))
            imga = imga/255.0
            # resizing the gray scale into 224x224, since we need a fixed common size for all the images in the dataset
            data.append(imga)
            target.append(label_dict[category])
            # appending the image and the label(categorized) into the list (dataset)

        except Exception as e:
            print('Exception:', e)
            # if any exception raised, the exception will be printed here. And pass to the next image


print(len(data))

data=np.array(data)
target=np.array(target)

print(data.shape)
print(target.shape)

np.save("D:/brainx_fold1.npy", data)
np.save("D:/brainy_fold1.npy", target)
