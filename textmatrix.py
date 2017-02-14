#!/usr/bin/env python 

import numpy as np
import pandas as pd
import itertools
import pprint
from sklearn import preprocessing

    input_file = '/home/james/Desktop/NIMH Data/hypericum/CDROM HYP PUB Ver 1/4 ASCII DATA/ael.csv'

def Csv
    #csv reader
    df = pd.read_csv(input_file, header = 0)

    original_headers = list(df.columns.values)
numeric_headers = list(df.columns.values)

numpy_array = df.as_matrix()

#SKLearn does not handle categorical variables by default. We need to use sklearn.preprocessing.OneHotEncoder
