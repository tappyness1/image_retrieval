import scipy
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def get_image_labels(labels_fpath='data/jpg/imagelabels.mat'):
    labels = scipy.io.loadmat(labels_fpath)
    img_labels = labels['labels'].ravel()
    img_label_dict = {f'{(i+1):05d}': img_labels[i] for i in range(len(img_labels))}
    return img_label_dict

def get_tpfp(input_img, img_list, img_label_dict):
    
    pred_class = []
    gt_class = [img_label_dict[input_img] for i in range(len(img_list))]
    
    for img in img_list: 
        split_1 = img.split(".")
        split_2 = split_1[0].split("_")
        pred_class.append(img_label_dict[split_2[1]])

    # make dataframe
    df = pd.DataFrame(list(zip(gt_class, pred_class)),
               columns =['Ground Truth', 'Prediction'])

    return df

def visualise(input_img, img_list, img_folder = "data/jpg/"):
    k = len(img_list)
    fig=plt.figure(figsize=(6,10))
    columns = 2
    print (columns)
    rows = int((k+2)/2)
    input_img = cv2.imread(input_img)
    input_img = input_img[...,::-1]
    input_img = cv2.resize(input_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    # plot_input_img = plt.imshow(input_img)
    fig.add_subplot(rows, columns, 1)
    plot_input_img = plt.imshow(input_img)
    for i in range(3, k+3):
        img = cv2.imread(f"{img_folder}{img_list[i-3]}")
        img = img[...,::-1]
        img = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)### what you want you can plot  
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    