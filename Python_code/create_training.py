#script to create kaggle style training data from folders of classified images

import os
import pandas as pd
from pandas import DataFrame

def create_csvData(fullDir):
    classes = [d for d in os.listdir(fullDir) if os.path.isdir(os.path.join(fullDir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    num_to_class = dict(zip(range(len(classes)), classes))

    train = []
    for index, label in enumerate(classes):
        path = fullDir + label + '/'
        for file in os.listdir(path):
            train.append(['{}/{}'.format(label, file), label, index])

    df = pd.DataFrame(train, columns=['file', 'category', 'category_id', ])

    return classes, class_to_idx, num_to_class, df

Dir = 'C:/Users/gkxmm/Images/'
data = create_csvData(Dir)
data = data[3]
data = DataFrame(data, columns=['file','category','category_id'])
export_csv = data.to_csv (r'C:/Users/gkxmm/Images/traindata.csv', index = None, header=True)
