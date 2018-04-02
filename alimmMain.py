import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import date
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from sklearn import preprocessing
import time
import warnings
warnings.filterwarnings("ignore")
datestr = date.today()
trainDataPath = 'train.txt'
testDataPath  = 'test.txt'
newTrainData_easy = 'newTrainInput_easy.csv'
newTrainData_hard = 'newTrainInput_hard.csv'
lgb_kf = 'lgb_kf.model'

def base_process(trainFile, testFile):
    """
     Desc：做训练输入数据和预测输入数据， trainFile: 训练数据 testFile: 测试数据
    """
    print(
        '--------------------------------------------------------------Data processing --------------------------------------------------------------' )

    train = pd.read_csv(trainFile, sep="\s+" )
    test = pd.read_csv( testFile, sep="\s+" )
    data = pd.concat( [train, test] )
    data = data.drop_duplicates( subset='instance_id' )  # 去重

    lbl = preprocessing.LabelEncoder()# 编码

    print(
        '--------------------------------------------------------------item--------------------------------------------------------------')
    data['len_item_category'] = data['item_category_list'].map(lambda x: len(str(x).split(';')))
    data['len_item_property'] = data['item_property_list'].map(lambda x: len(str(x).split(';')))
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for i in range(10):
        data['item_property_list' + str(i)] = lbl.fit_transform(data['item_property_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    for col in ['item_id', 'item_brand_id', 'item_city_id']:
        data[col] = lbl.fit_transform(data[col])

    print(
        '--------------------------------------------------------------user--------------------------------------------------------------')
    for col in ['user_id']:
        data[col] = lbl.fit_transform(data[col])
    print('user 0,1 feature')
    data['gender0'] = data['user_gender_id'].apply(lambda x: 1 if x == -1 else 2)
    data['age0'] = data['user_age_level'].apply(lambda x: 1 if x == 1004 | x == 1005 | x == 1006 | x == 1007  else 2)
    data['star0'] = data['user_star_level'].apply(lambda x: 1 if x == -1 | x == 3000 | x == 3001  else 2)
    data['occupation0'] = data['user_occupation_id'].apply(lambda x: 1 if x == -1 | x == 2003  else 2)

    print(
        '--------------------------------------------------------------context--------------------------------------------------------------')
    def timestamp_datetime(value):
        return time.strftime( '%Y-%m-%d %H:%M:%S', time.localtime( value ) )# 时间换为年月日，但有时区问题，Linux系统 差八小时
    data['realtime'] = data['context_timestamp'].apply(timestamp_datetime)
    data['realtime'] = pd.to_datetime(data['realtime'])
    data['day'] = data['realtime'].dt.day
    data['hour'] = data['realtime'].dt.hour
    data['len_predict_category_property'] = data['predict_category_property'].map(lambda x: len(str(x).split(';')))
    for i in range(5):
        data['predict_category_property' + str(i)] = lbl.fit_transform(data['predict_category_property'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    print('context 0,1 feature')
    data['context_page0'] = data['context_page_id'].apply(
        lambda x: 1 if x == 4001 | x == 4002 | x == 4003 | x == 4004 | x == 4007  else 2)

    def map_hour(x):
        if (x >= 7) & (x <= 12):
            return 1
        elif (x >= 13) & (x <= 20):
            return 2
        else:
            return 3
    data['hour_map'] = data['hour'].apply( map_hour )

    print( '当前日期前一天的cnt' )
    for d in range( 19, 26 ):  # 18到24号
        df1 = data[data['day'] == d - 1]
        df2 = data[data['day'] == d]  # 19到25号
        user_cnt = df1.groupby( by='user_id' ).count()['instance_id'].to_dict()
        item_cnt = df1.groupby( by='item_id' ).count()['instance_id'].to_dict()
        shop_cnt = df1.groupby( by='shop_id' ).count()['instance_id'].to_dict()
        df2['user_cnt1'] = df2['user_id'].apply( lambda x: user_cnt.get( x, 0 ) )
        df2['item_cnt1'] = df2['item_id'].apply( lambda x: item_cnt.get( x, 0 ) )
        df2['shop_cnt1'] = df2['shop_id'].apply( lambda x: shop_cnt.get( x, 0 ) )
        df2 = df2[['user_cnt1', 'item_cnt1', 'shop_cnt1', 'instance_id']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat( [df2, Df2] )
    data = pd.merge( data, Df2, on=['instance_id'], how='left' )
    print( '当前日期之前的cnt' )
    for d in range( 19, 26 ):
        # 19到25，25是test
        df1 = data[data['day'] < d]
        df2 = data[data['day'] == d]
        user_cnt = df1.groupby( by='user_id' ).count()['instance_id'].to_dict()
        item_cnt = df1.groupby( by='item_id' ).count()['instance_id'].to_dict()
        shop_cnt = df1.groupby( by='shop_id' ).count()['instance_id'].to_dict()
        df2['user_cntx'] = df2['user_id'].apply( lambda x: user_cnt.get( x, 0 ) )
        df2['item_cntx'] = df2['item_id'].apply( lambda x: item_cnt.get( x, 0 ) )
        df2['shop_cntx'] = df2['shop_id'].apply( lambda x: shop_cnt.get( x, 0 ) )
        df2 = df2[['user_cntx', 'item_cntx', 'shop_cntx', 'instance_id']]
        if d == 19:
            Df2 = df2
        else:
            Df2 = pd.concat( [df2, Df2] )
    data = pd.merge( data, Df2, on=['instance_id'], how='left' )
    print(
        '--------------------------------------------------------------shop--------------------------------------------------------------')
    for col in ['shop_id']:
        data[col] = lbl.fit_transform(data[col])
    data['shop_score_delivery0'] = data['shop_score_delivery'].apply(lambda x: 0 if x <= 0.98 and x >= 0.96  else 1)

    def deliver(x):
        # x=round(x,6)
        jiange = 0.1
        for i in range( 1, 20 ):
            if (x >= 4.1 + jiange * (i - 1)) & (x <= 4.1 + jiange * i):
                return i + 1
        if x == -5:
            return 1
    def deliver1(x):
        if (x >= 2) & (x <= 4):
            return 1
        elif (x >= 5) & (x <= 7):
            return 2
        else:
            return 3
    def review(x):
        # x=round(x,6)
        jiange = 0.02
        for i in range( 1, 30 ):
            if (x >= 0.714 + jiange * (i - 1)) & (x <= 0.714 + jiange * i):
                return i + 1
        if x == -1:
            return 1
    def review1(x):
        # x=round(x,6)
        if (x >= 2) & (x <= 12):
            return 1
        elif (x >= 13) & (x <= 15):
            return 2
        else:
            return 3
    def service(x):
        # x=round(x,6)
        jiange = 0.1
        for i in range( 1, 20 ):
            if (x >= 3.93 + jiange * (i - 1)) & (x <= 3.93 + jiange * i):
                return i + 1
        if x == -1:
            return 1
    def service1(x):
        if (x >= 2) & (x <= 7):
            return 1
        elif (x >= 8) & (x <= 9):
            return 2
        else:
            return 3
    def describe(x):
        # x=round(x,6)
        jiange = 0.1
        for i in range( 1, 30 ):
            if (x >= 3.93 + jiange * (i - 1)) & (x <= 3.93 + jiange * i):
                return i + 1
        if x == -1:
            return 1
    def describe1(x):
        if (x >= 2) & (x <= 8):
            return 1
        elif (x >= 9) & (x <= 10):
            return 2
        else:
            return 3

    data['shop_score_delivery'] = data['shop_score_delivery'] * 5
    data = data[data['shop_score_delivery'] != -5]
    data['deliver_map'] = data['shop_score_delivery'].apply( deliver )
    data['deliver_map'] = data['deliver_map'].apply( deliver1 )
    # del data['shop_score_delivery']
    print( data.deliver_map.value_counts() )

    data['shop_score_service'] = data['shop_score_service'] * 5
    data = data[data['shop_score_service'] != -5]
    data['service_map'] = data['shop_score_service'].apply( service )
    data['service_map'] = data['service_map'].apply( service1 )
    # del data['shop_score_service']
    print( data.service_map.value_counts() )  # 视为好评，中评，差评
    #
    data['shop_score_description'] = data['shop_score_description'] * 5
    data = data[data['shop_score_description'] != -5]
    data['de_map'] = data['shop_score_description'].apply( describe )
    data['de_map'] = data['de_map'].apply( describe1 )
    # del data['shop_score_description']
    print( data.de_map.value_counts() )

    data = data[data['shop_review_positive_rate'] != -1]
    data['review_map'] = data['shop_review_positive_rate'].apply( review )
    data['review_map'] = data['review_map'].apply( review1 )
    print( data.review_map.value_counts() )

    data['normal_shop'] = data.apply(
        lambda x: 1 if (x.deliver_map == 3) & (x.service_map == 3) & (x.de_map == 3) & (x.review_map == 3) else 0,
        axis=1 )
    del data['de_map']
    del data['service_map']
    del data['deliver_map']
    del data['review_map']

    data.to_csv( 'newTrainInput_easy.csv', index=False )

    return data
def makeMergeInput_hard( dataFile ):
    """
    Desc：把数据做组合，关联 dataFile: newTrainInput_easy.csv 是已经被切割和 easy 处理过的数据
    """
    print(
        '--------------------------------------------------------------Merge Data --------------------------------------------------------------' )
    data = pd.read_csv( dataFile )

    def zuhe(data):
        for col in ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
            data[col] = data[col].apply( lambda x: 0 if x == -1 else x )

        for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                    'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                    'shop_review_num_level', 'shop_star_level']:
            data[col] = data[col].astype( str )

        print( 'item两两组合' )
        data['sale_price'] = data['item_sales_level'] + data['item_price_level']
        data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
        data['price_collect'] = data['item_price_level'] + data['item_collected_level']

        print( 'user两两组合' )
        data['gender_age'] = data['user_gender_id'] + data['user_age_level']
        data['gender_occ'] = data['user_gender_id'] + data['user_occupation_id']
        data['gender_star'] = data['user_gender_id'] + data['user_star_level']

        print( 'shop两两组合' )
        data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']

        for col in ['item_sales_level', 'item_price_level', 'item_collected_level', 'sale_price', 'sale_collect',
                    'price_collect',
                    'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'gender_age',
                    'gender_occ', 'gender_star',
                    'shop_review_num_level', 'shop_star_level', 'review_star']:
            data[col] = data[col].astype( int )

        del data['review_star']

        return data
    def item(data):
        print( '一个item有多少brand,price salse collected level……' )

        itemcnt = data.groupby( ['item_id'], as_index=False )['instance_id'].agg( {'item_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['item_id'], how='left' )

        for col in ['item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level',
                    'item_pv_level']:
            itemcnt = data.groupby( [col, 'item_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_item_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'item_id'], how='left' )
            data[str( col ) + '_item_prob'] = data[str( col ) + '_item_cnt'] / data['item_cnt']
        del data['item_cnt']

        print( '一个brand有多少price salse collected level……' )

        itemcnt = data.groupby( ['item_brand_id'], as_index=False )['instance_id'].agg( {'item_brand_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['item_brand_id'], how='left' )

        for col in ['item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
            itemcnt = data.groupby( [col, 'item_brand_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_brand_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'item_brand_id'], how='left' )
            data[str( col ) + '_brand_prob'] = data[str( col ) + '_brand_cnt'] / data['item_brand_cnt']
        del data['item_brand_cnt']

        print( '一个city有多少item_price_level，item_sales_level，item_collected_level，item_pv_level' )

        itemcnt = data.groupby( ['item_city_id'], as_index=False )['instance_id'].agg( {'item_city_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['item_city_id'], how='left' )
        for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
            itemcnt = data.groupby( [col, 'item_city_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_city_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'item_city_id'], how='left' )
            data[str( col ) + '_city_prob'] = data[str( col ) + '_city_cnt'] / data['item_city_cnt']
        del data['item_city_cnt']

        print( '一个price有多少item_sales_level，item_collected_level，item_pv_level' )

        itemcnt = data.groupby( ['item_price_level'], as_index=False )['instance_id'].agg( {'item_price_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['item_price_level'], how='left' )
        for col in ['item_sales_level', 'item_collected_level', 'item_pv_level']:
            itemcnt = data.groupby( [col, 'item_city_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_price_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'item_city_id'], how='left' )
            data[str( col ) + '_price_prob'] = data[str( col ) + '_price_cnt'] / data['item_price_cnt']
        del data['item_price_cnt']

        print( '一个item_sales_level有多少item_collected_level，item_pv_level' )

        itemcnt = data.groupby( ['item_sales_level'], as_index=False )['instance_id'].agg( {'item_salse_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['item_sales_level'], how='left' )
        for col in ['item_collected_level', 'item_pv_level']:
            itemcnt = data.groupby( [col, 'item_sales_level'], as_index=False )['instance_id'].agg(
                {str( col ) + '_salse_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'item_sales_level'], how='left' )
            data[str( col ) + '_salse_prob'] = data[str( col ) + '_salse_cnt'] / data['item_salse_cnt']
        del data['item_salse_cnt']

        print( '一个item_collected_level有多少item_pv_level' )

        itemcnt = data.groupby( ['item_collected_level'], as_index=False )['instance_id'].agg(
            {'item_coll_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['item_collected_level'], how='left' )
        for col in ['item_pv_level']:
            itemcnt = data.groupby( [col, 'item_collected_level'], as_index=False )['instance_id'].agg(
                {str( col ) + '_coll_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'item_collected_level'], how='left' )
            data[str( col ) + '_coll_prob'] = data[str( col ) + '_coll_cnt'] / data['item_coll_cnt']
        del data['item_coll_cnt']

        return data
    def user(data):
        print( '用户有多少性别' )
        itemcnt = data.groupby( ['user_id'], as_index=False )['instance_id'].agg( {'user_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['user_id'], how='left' )

        for col in ['user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level']:
            itemcnt = data.groupby( [col, 'user_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'user_id'], how='left' )
            data[str( col ) + '_user_prob'] = data[str( col ) + '_user_cnt'] / data['user_cnt']
        del data['user_cnt']

        print( '性别的年龄段，职业有多少' )
        itemcnt = data.groupby( ['user_gender_id'], as_index=False )['instance_id'].agg( {'user_gender_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['user_gender_id'], how='left' )

        for col in ['user_age_level', 'user_occupation_id', 'user_star_level']:
            itemcnt = data.groupby( [col, 'user_gender_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_gender_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'user_gender_id'], how='left' )
            data[str( col ) + '_user_gender_prob'] = data[str( col ) + '_user_gender_cnt'] / data['user_gender_cnt']
        del data['user_gender_cnt']

        print( 'user_age_level对应的user_occupation_id，user_star_level' )
        itemcnt = data.groupby( ['user_age_level'], as_index=False )['instance_id'].agg( {'user_age_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['user_age_level'], how='left' )

        for col in ['user_occupation_id', 'user_star_level']:
            itemcnt = data.groupby( [col, 'user_age_level'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_age_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'user_age_level'], how='left' )
            data[str( col ) + '_user_age_prob'] = data[str( col ) + '_user_age_cnt'] / data['user_age_cnt']
        del data['user_age_cnt']

        print( 'user_occupation_id对应的user_star_level' )
        itemcnt = data.groupby( ['user_occupation_id'], as_index=False )['instance_id'].agg( {'user_occ_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['user_occupation_id'], how='left' )
        for col in ['user_star_level']:
            itemcnt = data.groupby( [col, 'user_occupation_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_occ_cnt': 'count'} )
            data = pd.merge( data, itemcnt, on=[col, 'user_occupation_id'], how='left' )
            data[str( col ) + '_user_occ_prob'] = data[str( col ) + '_user_occ_cnt'] / data['user_occ_cnt']
        del data['user_occ_cnt']

        return data
    def user_item(data):
        itemcnt = data.groupby( ['user_id'], as_index=False )['instance_id'].agg( {'user_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['user_id'], how='left' )
        print( '一个user有多少item_id,item_brand_id……' )
        for col in ['item_id',
                    'item_brand_id', 'item_city_id', 'item_price_level',
                    'item_sales_level', 'item_collected_level', 'item_pv_level']:
            item_shop_cnt = data.groupby( [col, 'user_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'user_id'], how='left' )
            data[str( col ) + '_user_prob'] = data[str( col ) + '_user_cnt'] / data['user_cnt']

        print( '一个user_gender有多少item_id,item_brand_id……' )
        itemcnt = data.groupby( ['user_gender_id'], as_index=False )['instance_id'].agg( {'user_gender_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['user_gender_id'], how='left' )
        for col in ['item_id',
                    'item_brand_id', 'item_city_id', 'item_price_level',
                    'item_sales_level', 'item_collected_level', 'item_pv_level']:
            item_shop_cnt = data.groupby( [col, 'user_gender_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_gender_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'user_gender_id'], how='left' )
            data[str( col ) + '_user_gender_prob'] = data[str( col ) + '_user_gender_cnt'] / data['user_gender_cnt']

        print( '一个user_age_level有多少item_id,item_brand_id……' )
        itemcnt = data.groupby( ['user_age_level'], as_index=False )['instance_id'].agg( {'user_age_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['user_age_level'], how='left' )
        for col in ['item_id',
                    'item_brand_id', 'item_city_id', 'item_price_level',
                    'item_sales_level', 'item_collected_level', 'item_pv_level']:
            item_shop_cnt = data.groupby( [col, 'user_age_level'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_age_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'user_age_level'], how='left' )
            data[str( col ) + '_user_age_prob'] = data[str( col ) + '_user_age_cnt'] / data['user_age_cnt']

        print( '一个user_occupation_id有多少item_id,item_brand_id…' )
        itemcnt = data.groupby( ['user_occupation_id'], as_index=False )['instance_id'].agg( {'user_occ_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['user_occupation_id'], how='left' )
        for col in ['item_id',
                    'item_brand_id', 'item_city_id', 'item_price_level',
                    'item_sales_level', 'item_collected_level', 'item_pv_level']:
            item_shop_cnt = data.groupby( [col, 'user_occupation_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_occ_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left' )
            data[str( col ) + '_user_occ_prob'] = data[str( col ) + '_user_occ_cnt'] / data['user_occ_cnt']

        return data
    def user_shop(data):
        print( '一个user有多少shop_id,shop_review_num_level……' )

        for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
            item_shop_cnt = data.groupby( [col, 'user_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'user_id'], how='left' )
            data[str( col ) + '_user_prob'] = data[str( col ) + '_user_cnt'] / data['user_cnt']
        del data['user_cnt']

        print( '一个user_gender有多少shop_id,shop_review_num_level……' )
        for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
            item_shop_cnt = data.groupby( [col, 'user_gender_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_gender_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'user_gender_id'], how='left' )
            data[str( col ) + '_user_gender_prob'] = data[str( col ) + '_user_gender_cnt'] / data['user_gender_cnt']
        del data['user_gender_cnt']

        print( '一个user_age_level有多少shop_id,shop_review_num_level……' )
        for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
            item_shop_cnt = data.groupby( [col, 'user_age_level'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_age_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'user_age_level'], how='left' )
            data[str( col ) + '_user_age_prob'] = data[str( col ) + '_user_age_cnt'] / data['user_age_cnt']
        del data['user_age_cnt']

        print( '一个user_occupation_id有多少shop_id,shop_review_num_level……' )
        for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
            item_shop_cnt = data.groupby( [col, 'user_occupation_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_user_occ_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left' )
            data[str( col ) + '_user_occ_prob'] = data[str( col ) + '_user_occ_cnt'] / data['user_occ_cnt']
        del data['user_occ_cnt']

        return data
    def shop_item(data):
        print( '一个shop有多少item_id,item_brand_id,item_city_id,item_price_level……' )
        itemcnt = data.groupby( ['shop_id'], as_index=False )['instance_id'].agg( {'shop_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['shop_id'], how='left' )
        for col in ['item_id',
                    'item_brand_id', 'item_city_id', 'item_price_level',
                    'item_sales_level', 'item_collected_level', 'item_pv_level']:
            item_shop_cnt = data.groupby( [col, 'shop_id'], as_index=False )['instance_id'].agg(
                {str( col ) + '_shop_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'shop_id'], how='left' )
            data[str( col ) + '_shop_prob'] = data[str( col ) + '_shop_cnt'] / data['shop_cnt']
        del data['shop_cnt']

        print( '一个shop_review_num_level有多少item_id,item_brand_id,item_city_id,item_price_level……' )
        itemcnt = data.groupby( ['shop_review_num_level'], as_index=False )['instance_id'].agg(
            {'shop_rev_cnt': 'count'} )
        data = pd.merge( data, itemcnt, on=['shop_review_num_level'], how='left' )
        for col in ['item_id',
                    'item_brand_id', 'item_city_id', 'item_price_level',
                    'item_sales_level', 'item_collected_level', 'item_pv_level']:
            item_shop_cnt = data.groupby( [col, 'shop_review_num_level'], as_index=False )['instance_id'].agg(
                {str( col ) + '_shop_rev_cnt': 'count'} )
            data = pd.merge( data, item_shop_cnt, on=[col, 'shop_review_num_level'], how='left' )
            data[str( col ) + '_shop_rev_prob'] = data[str( col ) + '_shop_rev_cnt'] / data['shop_rev_cnt']
        del data['shop_rev_cnt']

        return data

    data = zuhe(data)
    data = item( data )
    data = user( data )
    data = user_item( data )
    data = user_shop( data )
    data = shop_item( data )
    data.to_csv( 'newTrainInput_hard.csv', index=False )

def lgbTrainModel( dataFile ):
    """
    Desc：用lightgbm来跑出模型，dataFile: easy or hard 可以选择 read.csv(xxx) 来测不同的数据
            xTrain, yTrain 和 xTest, yTest 的策略选取很重要。多思考，多尝试
    """
    print(
        '--------------------------------------------------------------Traning model--------------------------------------------------------------' )
    data = pd.read_csv( dataFile )
    train = data[(data['day'] >= 18) & (data['day'] <= 23)]
    test = data[(data['day'] == 24)]
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]

    xTrain = train[col]
    yTrain = train['is_trade'].values
    xTest = test[col]
    yTest = test['is_trade'].values

    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        # depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=20000)
    model = lgb0.fit(xTrain, yTrain, eval_set=[(xTest, yTest)], early_stopping_rounds=1000)#rounds 可以设置大一点，看看跑的效果如何
    best_iter = model.best_iteration_
    predicted_score = model.predict_proba(test[col])[:, 1]

    joblib.dump( model, "pima.joblib01.dat" )# 用于存储训练出的模型
    #loaded_model = joblib.load("pima.joblib.dat")# 用于导出已经训练好的模型

    test['predicted_score'] = predicted_score# 将预测的概率值放入
    test['index'] = range(len(test))
    print(test[['is_trade','predicted_score']])
    print('线下误差------> ', log_loss(test['is_trade'], test['predicted_score']))# 比赛给出的误差公式

    test[['instance_id', 'predicted_score']].to_csv( '/Users/myy/Documents/AlimmResult/validtionData_trainingModel_result_{}.csv'.format(datestr), sep=" ", index=False )

    return best_iter
def lgbPredict(dataFile, best_iter):
    """
    Desc：模型预测，best_iter: 训练过程中得到的最佳迭代参数
            这里可以换一个模型，lgb.train() 来做training 完后，保存模型，直接拉出来用
    """
    print(
        '--------------------------------------------------------------Testing model--------------------------------------------------------------' )

    data = pd.read_csv( dataFile )
    train = data[data.is_trade.notnull()]
    test = data[data.is_trade.isnull()]
    col = [c for c in train if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]
    xTrain = train[col]
    yTrain = train['is_trade'].values

    lgb0 = lgb.LGBMClassifier(
        objective='binary',
        # metric='binary_error',
        num_leaves=35,
        # depth=8,
        learning_rate=0.05,
        seed=2018,
        colsample_bytree=0.8,
        # min_child_samples=8,
        subsample=0.9,
        n_estimators=best_iter)

    lgb_model = lgb0.fit(xTrain, yTrain)
    predicted_score = lgb_model.predict_proba(test[col])[:, 1]

    test['predicted_score'] = predicted_score
    sub1 = test[['instance_id', 'predicted_score']]
    sub=pd.read_csv("test.txt", sep="\s+")
    sub=pd.merge(sub,sub1,on=['instance_id'],how='left')
    sub=sub.fillna(0)

    sub[['instance_id', 'predicted_score']].to_csv('/Users/myy/Documents/AlimmResult/predictionData_trainingModel_result_{}.csv'.format(datestr),sep=" ",index=False)
    # sub[['instance_id', 'predicted_score']].to_csv('result.csv', sep=" ",index=False )# submit到天池的时候，是这个命名格式的文档


def lgbTrainModel2( dataFile ):
    """
    Desc：用lightgbm来跑出模型，dataFile: easy or hard 可以选择 read.csv(xxx) 来测不同的数据
            xTrain, yTrain 和 xTest, yTest 的策略选取很重要。多思考，多尝试
    """
    print(
        '--------------------------------------------------------------Traning model--------------------------------------------------------------' )
    data = pd.read_csv( dataFile )
    col = [c for c in data if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]

    trainInput = data[col]
    isTrade = data['is_trade'].values

    xTrain, xTest, yTrain, yTest = train_test_split( trainInput, isTrade, test_size=0.2, random_state=100 )
    lgbTrain = lgb.Dataset( xTrain, yTrain )
    lgbEval = lgb.Dataset( xTest, yTest )
    lgbAll = lgb.Dataset( trainInput, isTrade )

    # 使用GridSearchCV调lgb的参
    param_grid = {
        'learning_rate': [0.3, 0.4, 0.5],
        'num_leaves': [30, 10, 20]
    }
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'seed': 0,
        'num_leaves': 30,
        'learning_rate': 0.05,
    }
    numRound = 10  # 不会过拟合的情况下，可以设大一点
    modelTrain = lgb.train( params, lgbTrain, numRound, valid_sets=lgbEval, early_stopping_rounds=15 )
    # 用分出的部分训练集测出的最佳迭代次数在，全体训练集中重新训练
    model = lgb.train( params, lgbAll, modelTrain.best_iteration )
    model.save_model( 'lgb01.model' )  # 用于存储训练出的模型

    print(model.feature_importance()) # 看lgb模型特征得分
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = model.feature_name()# 看各个feature 所占权重
    dfFeature['weight'] = model.feature_importance()
    dfFeature.to_csv( 'model_2_featureImportance01.csv' )
    # return best_iter

    predTest = model.predict( trainInput )
    print( '模型 lgb_2 误差----> : ', log_loss( isTrade, predTest ) )

def lgbKFoldTrainModel(dataFile):
    """
    Desc：用lightgbm来跑出模型，dataFile: easy or hard 可以选择 read.csv(xxx) 来测不同的数据
                xTrain, yTrain 和 xTest, yTest 的策略选取很重要。多思考，多尝试
    """
    print(
        '--------------------------------------------------------------Traning KF model--------------------------------------------------------------' )
    data = pd.read_csv( dataFile )
    col = [c for c in data if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]


    trainInput = data[col]
    isTrade = data['is_trade']

    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'seed': 0,
        'num_leaves': 30,
        'learning_rate': 0.05,
    }

    lgbAll = lgb.Dataset( trainInput, isTrade )
    # KFold交叉验证
    kf = KFold( n_splits=5, random_state=0 )
    kf.get_n_splits( trainInput )
    print( kf )
    bestIterRecord = []  # 记录每次的最佳迭代次数
    rmseRecord = []  # 记录每次的最佳迭代点的rmse
    numRound = 10  # 不会过拟合的情况下，可以设大一点
    for trainIndex, testIndex in kf.split( trainInput ):
        print( "Train Index:", trainIndex, ",Test Index:", testIndex )
        xTrain, xTest = trainInput.iloc[trainIndex], trainInput.iloc[testIndex]
        yTrain, yTest = isTrade.iloc[trainIndex], isTrade.iloc[testIndex]

        lgbTrain = lgb.Dataset( xTrain, yTrain )
        lgbEval = lgb.Dataset( xTest, yTest )
        evalRmse = {}  # 存储实时的rmse结果
        modelTrain = lgb.train(
            params=params,
            train_set=lgbTrain,
            num_boost_round=numRound,
            valid_sets=lgbEval,
            valid_names='get_rmse',
            evals_result=evalRmse,
            early_stopping_rounds=15 )

        bestIterRecord.append( modelTrain.best_iteration )
        rmseRecord.append( evalRmse.get( 'get_rmse' ).get( 'rmse' )[modelTrain.best_iteration - 1] )

    bestIter = int( np.mean( bestIterRecord ) )  # 利用KFold求出的平均最佳迭代次数

    # 用分出的部分训练集测出的最佳迭代次数在，全体训练集中重新训练
    model = lgb.train( params, lgbAll, bestIter )
    model.save_model( 'lgb_kf.model' )  # 用于存储训练出的模型
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = model.feature_name()
    dfFeature['score'] = model.feature_importance()
    dfFeature.to_csv( 'model_KF_featureImportance.csv' )

    predTest = model.predict( trainInput )
    print( 'mean of rmse : ', np.mean( rmseRecord ) )
    print( 'best iteration : ', bestIter )
    print( '模型 lgb_kf 误差----> : ', log_loss(isTrade, predTest))

    return bestIter
def lgbKFoldPredict(datFile, modelFile):
    """
    Desc：借助已经跑出的模型
    """
    model = lgb.Booster(model_file = modelFile) #init model

    data = pd.read_csv(datFile)
    col = [c for c in data if
           c not in ['is_trade', 'item_category_list', 'item_property_list', 'predict_category_property', 'instance_id',
                     'context_id', 'realtime', 'context_timestamp']]

    preds = model.predict(data[col])
    instance_id = data['instance_id']  # 一个中括号是series类型，两个中括号是DataFrame类型
    res = pd.DataFrame()
    res['instance_id'] = instance_id
    res['predicted_score'] = preds
    res.to_csv('/Users/myy/Documents/AlimmResult/predictionData_lgbKFmodel_result_{}.csv'.format(datestr), index = False, encoding = 'utf-8', header = False)
    # res.to_csv( 'result.csv', index=False, encoding='utf-8', header=False )# submit到天池的时候，是这个命名格式的文档


if __name__ == "__main__":
    startTime = time.time()
    # data = base_process(trainDataPath, testDataPath)# 数据处理
    # makeMergeInput_hard(newTrainData_easy)# 构建关系feature engineering

    print(' ----------------------------------------------------------------Model 1--------------------------------------------------')
    best_iter = lgbTrainModel( newTrainData_hard )# 模型训练--easy 数据，找到最好的迭代，讲道理，模型应该换成 其他的
    lgbPredict( newTrainData_hard, best_iter )# 模型预测-- easy 数据

    # print('----------------------------------------------------------------Model 2--------------------------------------------------')
    # lgbTrainModel2( newTrainData_easy )# lgb模型， 但选用的 训练数据，和训练方式不一样，可以用这个代码块做feature 概率分别
    # # best_iter = lgbKFoldTrainModel( newTrainData_easy )# 将KF 和lgb 模型，目前这个模型效果最好
    # # lgbKFoldPredict(newTrainData_easy, lgb_kf)
    #
    # print(' ----------------------------------------------------------------Model 3--------------------------------------------------')
    # best_iter = lgbKFoldTrainModel( newTrainData_easy )
    # lgbPredict( newTrainData_easy, best_iter )  # 模型预测-- easy 数据

    print( "cost time:", time.time() - startTime, "(s)......" )