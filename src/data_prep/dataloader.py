# from src.main import image_retrieval

import scipy.io
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def get_df(labels_fpath = 'data/jpg/imagelabels.mat'):
    """Takes in the labels mat file and converts it to the anchor/positive/negative class images with original label

    Args:
        labels_fpath (str, optional): _description_. Defaults to 'data/jpg/imagelabels.mat'.

    Returns:
        _type_: _description_
    """

    labels = scipy.io.loadmat(labels_fpath)
    img_labels = labels['labels'].ravel()
    img_label_dict = {f'{(i+1):05d}': img_labels[i] for i in range(len(img_labels))}
    
    # print(img_label_dict)
    img_label_df = pd.DataFrame.from_dict(img_label_dict, orient='index', columns = ['label'])
    img_label_df = img_label_df.reset_index(names='anchor')

    # img_label_df.head()
    
    def get_negative(row):
        """helper function to get the negative column

        Args:
            row (_type_): _description_

        Returns:
            _type_: _description_
        """

        neg_candidates = img_label_df[img_label_df['label'] != row['label']]
        neg_candidates = neg_candidates[neg_candidates['anchor'] != row['anchor']]
        return neg_candidates.iloc[random.randint(0, neg_candidates.shape[0]-1)]['anchor']

    def get_positive(row):
        """helper function to get the positive column for output df

        Args:
            row (_type_): _description_

        Returns:
            _type_: _description_
        """

        pos_candidates = img_label_df[img_label_df['label'] == row['label']]
        # remove duplicate
        pos_candidates = pos_candidates[pos_candidates['anchor'] != row['anchor']]
        return pos_candidates.iloc[random.randint(0, pos_candidates.shape[0]-1)]['anchor']

    img_label_df['positive'] = img_label_df.apply(get_positive, axis = 1)
    img_label_df['negative'] = img_label_df.apply(get_negative, axis = 1)
    output_df = img_label_df[['anchor', 'positive', 'negative', 'label']]

    return output_df

def get_tvt_split(output_df):

    """with the output df, run a TVT Split

    Returns:
        _type_: _description_
    """

    train, validation = train_test_split(output_df, test_size=0.2)
    validation, test = train_test_split(validation, test_size = 0.5)

    return (train, validation, test)

def get_data_dict(tvts_tuple):

    """convert back to dictionary for easier digestion

    Returns:
        _type_: _description_
    """

    return [df.to_dict('index') for df in tvts_tuple]

def dataloader(labels_fpath = 'data/jpg/imagelabels.mat'):
    """full workflow from above

    Args:
        labels_fpath (str, optional): _description_. Defaults to 'data/jpg/imagelabels.mat'.

    Returns:
        data_dicts: a list of the train, validation and test dicts
    """
    output_df = get_df(labels_fpath)
    tvts_tuple = get_tvt_split(output_df)
    data_dicts = get_data_dict(tvts_tuple)
    return data_dicts
    
if __name__ == '__main__':
    output_df = get_df()
    tvts_tuple = get_tvt_split(output_df)
    data_dicts = get_data_dict(tvts_tuple)
    print (data_dicts)