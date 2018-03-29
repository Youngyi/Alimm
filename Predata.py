import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

class Predata:
    def time2cov(self, value):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(value))

    def trainData(self, train):
        print("***********training***********")
        for i in range(3):
            train['category_%d'%(i)] = train['item_category_list'].apply(lambda x: x.split(";")[i] if len(x.split(";"))> i else " ")
        del train['item_category_list']
        for i in range(3):
            train['property_%d'%(i)] = train['item_property_list'].apply(lambda x: x.split(";")[i] if len(x.split(";"))>i else " ")
        del train['item_property_list']

        train['context_timestamp'] = train['context_timestamp'].apply(self.time2cov)

        for i in range(3):
            train['predict_category_%d'%(i)] = train['predict_category_property'].apply(lambda x: str(x.split(";")[i] ).split(":")[0] if len(x.split(";"))>i else " ")

        return train


    def testData(self, test):
        print("***********testing************")
        for i in range( 3 ):
            test['category_%d' % (i)] = test['item_category_list'].apply(
                lambda x: x.split( ";" )[i] if len( x.split( ";" ) ) > i else " " )
        del test['item_category_list']
        for i in range( 3 ):
            test['property_%d' % (i)] = test['item_property_list'].apply(
                lambda x: x.split( ";" )[i] if len( x.split( ";" ) ) > i else " " )
        del test['item_property_list']

        test['context_timestamp'] = test['context_timestamp'].apply( self.time2cov )

        for i in range( 3 ):
            test['predict_category_%d' % (i)] = test['predict_category_property'].apply(
                lambda x: str( x.split( ";" )[i] ).split( ":" )[0] if len( x.split( ";" ) ) > i else " " )

        return test
    # def prepocessData(self, train):

ob = Predata()
train = pd.read_csv('train.txt', sep="\s+")
train = ob.trainData(train)
test = pd.read_csv('test.txt', sep="\s+")
test = ob.testData(test)

val = train[train['context_timestamp']>'2018-09-22 23:59:59']
train = train[train['context_timestamp']<='2018-09-21 23:59:59']
train = train[train['context_timestamp']>'2018-09-19 23:59:59']

y_train = train.pop('is_trade')
train_index = train.pop('instance_id')

y_val = val.pop('is_trade')
val_index = val.pop('instance_id')
test_index = test.pop('instance_id')

del train['context_timestamp']
del val['context_timestamp']
del test['context_timestamp']

from sklearn.preprocessing import LabelEncoder
# 对数据进行类别处理
all_data = pd.concat([train,val],axis=0)
all_data = pd.concat([all_data,test],axis=0)
# print(all_data.shape)

all_data = all_data[list(all_data.columns)].apply(LabelEncoder().fit_transform)

train = all_data[:train.shape[0]]
val = all_data[train.shape[0]:train.shape[0] + val.shape[0]]
test = all_data[train.shape[0] + val.shape[0]:]

del all_data