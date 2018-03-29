from Predata import train,np,val,test,y_train,y_val
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.models import Model
from keras.layers import Dense, Activation
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")
embeding_set = list(train.columns)
def get_mxlen_set(train,val,test,embeding_set=embeding_set):
    X = {}
    for ebd in embeding_set:
        X[ebd] = np.max([train[ebd].max(),val[ebd].max(),test[ebd].max()])+1
    return X
def get_kears_data(data,embeding_set=embeding_set):
    X = {}
    for ebd in embeding_set:
        X[ebd] = np.array(data[ebd])
    return X
def make_model():
    # 这个部分应该可以动态添加不用手动一个一个处理应该
    in_shop_score_description = Input(shape=[1], name='shop_score_description')
    shop_score_description = Embedding(mxlen_set['shop_score_description'], emb_n)(in_shop_score_description)

    in_user_id = Input(shape=[1], name='user_id')
    user_id = Embedding(mxlen_set['user_id'], emb_n)(in_user_id)

    in_user_age_level = Input(shape=[1], name='user_age_level')
    user_age_level = Embedding(mxlen_set['user_age_level'], emb_n)(in_user_age_level)

    in_shop_id = Input(shape=[1], name='shop_id')
    shop_id = Embedding(mxlen_set['shop_id'], emb_n)(in_shop_id)

    in_user_star_level = Input(shape=[1], name='user_star_level')
    user_star_level = Embedding(mxlen_set['user_star_level'], emb_n)(in_user_star_level)

    in_context_page_id = Input(shape=[1], name='context_page_id')
    context_page_id = Embedding(mxlen_set['context_page_id'], emb_n)(in_context_page_id)

    in_property_2 = Input(shape=[1], name='property_2')
    property_2 = Embedding(mxlen_set['property_2'], emb_n)(in_property_2)

    in_user_gender_id = Input(shape=[1], name='user_gender_id')
    user_gender_id = Embedding(mxlen_set['user_gender_id'], emb_n)(in_user_gender_id)

    in_shop_score_service = Input(shape=[1], name='shop_score_service')
    shop_score_service = Embedding(mxlen_set['shop_score_service'], emb_n)(in_shop_score_service)

    in_category_0 = Input(shape=[1], name='category_0')
    category_0 = Embedding(mxlen_set['category_0'], emb_n)(in_category_0)

    in_shop_review_positive_rate = Input(shape=[1], name='shop_review_positive_rate')
    shop_review_positive_rate = Embedding(mxlen_set['shop_review_positive_rate'], emb_n)(in_shop_review_positive_rate)

    in_context_id = Input(shape=[1], name='context_id')
    context_id = Embedding(mxlen_set['context_id'], emb_n)(in_context_id)

    in_predict_category_1 = Input(shape=[1], name='predict_category_1')
    predict_category_1 = Embedding(mxlen_set['predict_category_1'], emb_n)(in_predict_category_1)

    in_item_sales_level = Input(shape=[1], name='item_sales_level')
    item_sales_level = Embedding(mxlen_set['item_sales_level'], emb_n)(in_item_sales_level)

    in_item_brand_id = Input(shape=[1], name='item_brand_id')
    item_brand_id = Embedding(mxlen_set['item_brand_id'], emb_n)(in_item_brand_id)

    in_property_0 = Input(shape=[1], name='property_0')
    property_0 = Embedding(mxlen_set['property_0'], emb_n)(in_property_0)

    in_category_1 = Input(shape=[1], name='category_1')
    category_1 = Embedding(mxlen_set['category_1'], emb_n)(in_category_1)

    in_item_id = Input(shape=[1], name='item_id')
    item_id = Embedding(mxlen_set['item_id'], emb_n)(in_item_id)

    in_shop_star_level = Input(shape=[1], name='shop_star_level')
    shop_star_level = Embedding(mxlen_set['shop_star_level'], emb_n)(in_shop_star_level)

    in_category_2 = Input(shape=[1], name='category_2')
    category_2 = Embedding(mxlen_set['category_2'], emb_n)(in_category_2)

    in_item_price_level = Input(shape=[1], name='item_price_level')
    item_price_level = Embedding(mxlen_set['item_price_level'], emb_n)(in_item_price_level)

    in_predict_category_property = Input(shape=[1], name='predict_category_property')
    predict_category_property = Embedding(mxlen_set['predict_category_property'], emb_n)(in_predict_category_property)

    in_item_city_id = Input(shape=[1], name='item_city_id')
    item_city_id = Embedding(mxlen_set['item_city_id'], emb_n)(in_item_city_id)

    in_property_1 = Input(shape=[1], name='property_1')
    property_1 = Embedding(mxlen_set['property_1'], emb_n)(in_property_1)

    in_predict_category_2 = Input(shape=[1], name='predict_category_2')
    predict_category_2 = Embedding(mxlen_set['predict_category_2'], emb_n)(in_predict_category_2)

    in_predict_category_0 = Input(shape=[1], name='predict_category_0')
    predict_category_0 = Embedding(mxlen_set['predict_category_0'], emb_n)(in_predict_category_0)

    in_shop_score_delivery = Input(shape=[1], name='shop_score_delivery')
    shop_score_delivery = Embedding(mxlen_set['shop_score_delivery'], emb_n)(in_shop_score_delivery)

    in_item_collected_level = Input(shape=[1], name='item_collected_level')
    item_collected_level = Embedding(mxlen_set['item_collected_level'], emb_n)(in_item_collected_level)

    in_user_occupation_id= Input(shape=[1], name='user_occupation_id')
    user_occupation_id = Embedding(mxlen_set['user_occupation_id'], emb_n)(in_user_occupation_id)

    in_item_pv_level = Input(shape=[1], name='item_pv_level')
    item_pv_level = Embedding(mxlen_set['item_pv_level'], emb_n)(in_item_pv_level)

    in_shop_review_num_level = Input(shape=[1], name='shop_review_num_level')
    shop_review_num_level = Embedding(mxlen_set['shop_review_num_level'], emb_n)(in_shop_review_num_level)

    inp = concatenate([(shop_score_description), (user_id), (user_age_level), (shop_id), (context_page_id),
                      (property_2), (user_gender_id), (user_star_level), (shop_score_service), (category_0),
                      (shop_review_positive_rate), (context_id), (predict_category_1), (item_sales_level), (item_brand_id),
                      (property_0), (category_1), (item_id), (shop_star_level), (category_2),
                      (item_price_level), (predict_category_property), (item_city_id), (property_1), (predict_category_2),
                      (predict_category_0), (shop_score_delivery), (item_collected_level), (user_occupation_id), (item_pv_level),
                      (shop_review_num_level)
                      ])
    s_dout = SpatialDropout1D(0.2)(inp)
    x = Flatten()(s_dout)
    x = Dropout(0.2)(Dense(512,activation='relu')(x))
    x = Dropout(0.2)(Dense(64,activation='relu')(x))
    outp = Dense(1,activation='sigmoid')(x)
    # 手动加层快累死了
    model = Model(inputs=[
        in_shop_score_description,
        in_user_id,
        in_user_age_level,
        in_shop_id,
        in_user_star_level,
        in_context_page_id,
        in_property_2,
        in_user_gender_id,
        in_shop_score_service,
        in_category_0,
        in_shop_review_positive_rate,
        in_context_id,
        in_predict_category_1,
        in_item_sales_level,
        in_item_brand_id,
        in_property_0,
        in_category_1,
        in_item_id,
        in_shop_star_level,
        in_category_2,
        in_item_price_level,
        in_predict_category_property,
        in_item_city_id,
        in_property_1,
        in_predict_category_2,
        in_predict_category_0,
        in_shop_score_delivery,
        in_item_collected_level,
        in_user_occupation_id,
        in_item_pv_level,
        in_shop_review_num_level

    ], outputs=outp)
    return model

if __name__ == "__main__":
    print( "***********modling***********" )
    mxlen_set = get_mxlen_set(train,val,test) # print(mxlen_set) # print('make_keras_data_set')
    train = get_kears_data(train)
    val = get_kears_data(val)
    test_a = get_kears_data(test)
    emb_n = 50
    model = make_model()
    print(model.summary())

    batch_size = 1024
    epochs = 1

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    val_pre = model.predict(val,batch_size=1024)[:,0]
    print(log_loss(y_val,val_pre))