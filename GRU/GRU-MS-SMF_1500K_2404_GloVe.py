<<<<<<< HEAD
# %% [markdown]
# The MIT License (MIT)
# 
# Copyright (c) 2021 NVIDIA CORPORATION
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# %% [markdown]
# # GRU with MultiStage Session-based Matrix Factorization head

# %%
import os, sys

VER = 44
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Sistemde hangi gpu'nun çalışacağını belirler
MAX_GPU_MEMORY_MB = 20000 #hafızayı aşmaması için ekledim 0410


TRAIN_WITH_TEST = True
# ONLY DO THIS MANY FOLDS
#DO_FOLDS = 5 Daha kısa sürede dönmesi için 1'e çektim
DO_FOLDS = 5
# MAKE SUBMISSION OR NOT
DO_TEST = True

# %%
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import cudf
import cupy
import gc
from sklearn.model_selection import GroupKFold
import wandb
import random
import keyboard
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, GRU
from tensorflow.keras.models import Model
pd.set_option('display.max_columns', None)


pd.__version__, cudf.__version__

# %% [markdown]
# # Metric Calculation

# %%
# Return top4 metric
# istest: flag to select if metric should be computed in 0:train, 1:test, -1:both
# pos: select which city to calculate the metric, 0: last, 1: last-1, 2:last-2 , -1: all
# the input `val` dataframe must contains the target `city_id` and the 4 recommendations as: rec0, res1, rec2 and rec3
    
def top4_metric( val, istest=0, pos=0 , target='city_id'):
    
    if istest>=0:
        val = val.loc[ (val.submission==0) & (val.istest == istest) ]
    else:
        val = val.loc[ (val.submission==0) ]

    if pos >= 0:
        top1 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec0']
        top2 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec1']
        top3 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec2']
        top4 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec3']
    else:
        top1 = val[target] == val['rec0']
        top2 = val[target] == val['rec1']
        top3 = val[target] == val['rec2']
        top4 = val[target] == val['rec3']
        
    return (top1|top2|top3|top4).mean()  

##Wandb Uygulaması
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="GRU-MS-SMF-LessData",
#     config= {
#             "TRAIN_BATCH_SIZE" : 512,
#             "WORKERS" : 8,
#             "LR" : 1e-3,
#             "EPOCHS" : 5,
#             "GRADIENT_ACCUMULATION" : 1,
#             "EMBEDDING_DIM" : 64,
#             "HIDDEN_DIM" :  1024,
#             "DROPOUT_RATE" : 0.4,
#             }
# )  

# %% [markdown]
# # Feature Engineering

# %%
#%%time
PATH = './'
#raw = cudf.read_csv('../../00_Data/train_and_test_2.csv')
raw = cudf.read_csv('00_Data/train_and_test_2.csv')
print(raw.shape)

# %%
raw['reverse'] = 0
raw.head()

# %%
reverse = raw.copy()
reverse['reverse'] = 1
reverse['utrip_id'] = reverse['utrip_id']+'_r'

# %%
reverse['icount'] = reverse['dcount'].copy()

# %%
reverse = reverse.sort_values(['utrip_id', 'icount'], ascending=False)
raw = cudf.concat([raw, reverse], axis=0)
raw['sorting'] = cupy.asarray(range(raw.shape[0]))
del reverse
gc.collect()

# %%
raw['utrip_id'+'_'], mp = raw['utrip_id'].factorize()

# %%
#%%time

raw['checkin'] = cudf.to_datetime(raw.checkin, format="%Y-%m-%d")
raw['checkout'] = cudf.to_datetime(raw.checkout, format="%Y-%m-%d")

# %%
#%%time

# ENGINEER MONTH, WEEKDAY, and LENGTH OF STAY FEATURES
raw['mn'] = raw.checkin.dt.month
raw['dy1'] = raw.checkin.dt.weekday
raw['dy2'] = raw.checkout.dt.weekday
raw['length'] = (raw.checkout - raw.checkin).dt.days

# ENGINEER WEEKEND AND SEASON
raw['day_name']= raw.checkin.dt.weekday
raw['isweekend']=raw['day_name'].isin([5,6]).astype('int8')


# %%
df_season = cudf.DataFrame({'mn': range(1,13), 'season': ([0]*3)+([1]*3)+([2]*3)+([3]*3)})

# %%
raw=raw.merge(df_season, how='left', on='mn')
raw = raw.sort_values(['sorting'], ascending=True)

# %%
raw.head()

# %%
def shift_feature(df, groupby_col, col, offset, nan=-1, colname=''):
    df[colname] = df[col].shift(offset)
    df.loc[df[groupby_col]!=df[groupby_col].shift(offset), colname] = nan

# %%
# WEIRD FEATURE IS DIFFERENT FOR REVERSE AND NORMAL
shift_feature(raw, 'utrip_id_', 'checkout', 1, None, 'checkout_lag1')
raw['gap'] = (raw.checkin - raw.checkout_lag1).dt.days.fillna(-1) 

# %%
#%%time

# REVERSE ICOUNT
raw['dcount'] = raw['N']-raw['icount']-1

# %%
raw.head()

# %%
# LABEL ENCODE CATEGORICAL
CATS = ['city_id','mn','dy1','dy2','length','device_class','affiliate_id','booker_country',
        'hotel_country','icount','dcount','gap','isweekend', 'season']
MAPS = []
for c in CATS:
    raw[c+'_'], mp = raw[c].factorize()
    MAPS.append(mp)
    print('created',c+'_')

# %%
LAGS = 5
EC = 400

# ENGINEER LAG FEATURES
for i in range(1,LAGS+1):
    shift_feature(raw, 'utrip_id_', 'city_id_', i, -1, f'city_id_lag{i}')

# %%
shift_feature(raw, 'utrip_id_', 'hotel_country_', i, -1, 'hotel_lag1')

# %%
# TOTAL DURATION AND FIRST CITY
tmpA = raw[raw['icount']==0][['utrip_id', 'checkout']]
tmpB = raw[raw['dcount']==0][['utrip_id', 'checkin']]
tmpC = tmpA.merge(tmpB, how='left', on='utrip_id')
tmpC['length_total'] = (tmpC['checkout'] - tmpC['checkin']).dt.days
raw = raw.merge(tmpC[['utrip_id', 'length_total']],on='utrip_id',how='left')

# %%
tmpD = raw[raw['dcount']==0][['utrip_id', 'city_id_']]
tmpD.columns = ['utrip_id', 'city_first']
raw = raw.merge(tmpD,on='utrip_id',how='left')

# %%
tmpA = raw[raw['dcount']==0][['utrip_id', 'checkout']]
tmpB = raw[raw['icount']==0][['utrip_id', 'checkin']]
tmpC = tmpA.merge(tmpB, how='left', on='utrip_id')
tmpC['length_total_r'] = (tmpC['checkout'] - tmpC['checkin']).dt.days
raw = raw.merge(tmpC[['utrip_id', 'length_total_r']],on='utrip_id',how='left')
raw.loc[raw['length_total']<0,'length_total'] = raw.loc[raw['length_total']<0,'length_total_r']

# %%
raw[raw['utrip_id'].isin(['999944_1','999944_1_r'])]

# %% [markdown]
# # Finish Feature Engineering

# %%
# CONVERT -1 TO 0
cols = [f'city_id_lag{i}' for i in range(LAGS,0,-1)]
for c in cols+['city_id_','city_first']+['hotel_lag1']:
    raw[c] += 1
    print(c)

# %%
# MOVE RARE CITIES INTO COMMON CLASS
RARE = 9*2 #x2 because data has doubled
tmp = raw.city_id_.value_counts()
idx = tmp.loc[tmp<=RARE].index.values
print('rare city ct under <=%i'%RARE, len(idx),'total cities =',len(tmp) )

mx = raw.city_id_.max()+1
for c in cols+['city_first']:
    raw.loc[raw[c].isin(idx),c] = mx

# %%
# VERIFY LAG LOOKS RIGHT
raw[['city_id_','city_id_','icount']+cols].head()

# %%
FEATURES = ['mn_', 'dy1_','dy2_','length_','device_class_',  'affiliate_id_', 'booker_country_',
       'hotel_lag1','icount','dcount','city_first','length_total','gap_','isweekend_', 'season_'] + cols
TARGET = ['city_id_']

# %% [markdown]
# **Note that the order should be ['city_id_lag3', 'city_id_lag2', 'city_id_lag1'] in the order of time for the sake of RNN**

# %%
cols

# %% [markdown]
# # GRU-MS-SMF 5 Fold Model

# %%
os.environ['TF_MEMORY_ALLOCATION'] = "0.7" # fraction of free memory

# %%
import tensorflow as tf
tf.__version__

# %%
gpus = tf.config.list_physical_devices('GPU')
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7) #hafıza hatası almamak için ekledim 0410
# %%
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*19)]
)

# %% [markdown]
# # Define Embeddings

# %%
# EMBEDDING INPUT SIZES
cts = []
for c in FEATURES:
    mx = raw[c].max()
    cts.append(mx+1)
    print(c,mx+1)

# %%
# EMBEDDING OUTPUT SIZES
#EC = 400
cts2 = [4,3,3,6,2,60,3,14,7,7,200,18,9,3,4]+[EC]*len(cols)

# %%
emb_map = {i:(j,k) for i,j,k in zip(FEATURES, cts, cts2)}
emb_map

# %%
# TARGET SOFTMAX OUTPUT SIZE
t_ct = raw['city_id_'].max()+1
t_ct

# %% [markdown]
# # Build Model

# %%
N_CITY = 50

class Linear(tf.keras.layers.Layer):
    def __init__(self, H, activation='relu'):
        super(Linear, self).__init__()
        self.dense = tf.keras.layers.Dense(H)
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class WeightedSum(tf.keras.layers.Layer):
    def __init__(self):
        super(WeightedSum, self).__init__(name='weighted_sum')

    def build(self, input_shape):
        self.w1 = self.add_weight(name='w1',shape=None,dtype=tf.float32,
                                 trainable=True)
        self.w2 = self.add_weight(name='w2',shape=None,dtype=tf.float32,
                                 trainable=True)
        
    def call(self, x1, x2, x3):
        w1 = tf.nn.sigmoid(self.w1)
        w2 = tf.nn.sigmoid(self.w2)
        x1 = tf.stop_gradient(x1)
        x2 = tf.stop_gradient(x2)
        x3 = tf.stop_gradient(x3)
        return (x1 + x2*w1 + x3*w2)/(1+w1+w2)
    
class EmbDotSoftMax(tf.keras.layers.Layer):
    def __init__(self):
        super(EmbDotSoftMax, self).__init__()
        self.d1 = tf.keras.layers.Dense(EC) 
    
    def call(self, x, top_city_emb, top_city_id, prob):
        emb_pred = self.d1(x) # B,EC
        emb_pred = tf.expand_dims(emb_pred, axis=1) #B,1,EC
        x = emb_pred*top_city_emb #B,N_CITY,EC
        x = tf.math.reduce_sum(x, axis=2) #B,N_CITY
        x = tf.nn.softmax(x) #B,N_CITY

        rowids = tf.range(0,tf.shape(x)[0]) # B
        rowids = tf.transpose(tf.tile([rowids],[N_CITY,1])) # B,N_CITY

        idx = tf.stack([rowids,top_city_id],axis=2) # B, N_CITY, 2
        idx = tf.cast(idx, tf.int32)
        prob = tf.scatter_nd(idx, x, tf.shape(prob)) + 1e-6
        return prob
    
glove_embedding_path = 'glove.6B.100d.txt'  # Örneğin, dosya yolu

# Veri setinizdeki metinsel özellikler için bir kelime dizini oluşturun
word_index = {"Türkiye": 1, "İtalya": 2, ...}  # Örnek olarak, gerçek verilerle doldurulmalıdır.

# GloVe embeddinglerini yükleme fonksiyonu
def load_glove_embeddings(embedding_path, word_index, embedding_dim):
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix

# Model oluşturma fonksiyonu
def build_model_with_glove_embedding(embedding_matrix):
    inp = Input(shape=(len(FEATURES),))
    embs = []
    
    for k, f in enumerate(FEATURES):
        if f in ["booker_country", "hotel_country"]:
            # Metinsel özellikler için GloVe embeddinglerini kullanın
            embedding_dim = 100  # GloVe embedding boyutu
            num_words = len(word_index) + 1  # Kelime dizinindeki kelime sayısı + 1
            emb_layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)
            embs.append(emb_layer(inp[:, k]))
        else:
            # Diğer özellikler için normal embedding katmanlarını kullanın
            i, j = emb_map[f]
            embs.append(Embedding(i, j)(inp[:, k]))
    
    x = Concatenate()(embs)
    xc = GRU(EC, activation='tanh')(x)
    
    x1 = Linear(512+256, 'relu')(x)
    x2 = Linear(512+256, 'relu')(x1)
    prob = Dense(t_ct, activation='softmax', name='main_output')(x2)
    
    _, top_city_id = tf.math.top_k(prob, N_CITY)
    top_city_emb = emb_map['city_id_lag1'](top_city_id)
    
    x1 = Linear(512+256, 'relu')(x1)
    prob_1 = EmbDotSoftMax()(x1, top_city_emb, top_city_id, prob)
    prob_2 = EmbDotSoftMax()(x2, top_city_emb, top_city_id, prob)
    
    prob_ws = WeightedSum()(prob, prob_1, prob_2)
    
    model = Model(inputs=inp, outputs=[prob, prob_1, prob_2, prob_ws])
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    mtr = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=4)
    model.compile(loss=loss, optimizer=opt, metrics=[mtr])
    return model

# %% [markdown]
# # Learning Schedule

# %%
# CUSTOM LEARNING SCHEUDLE

def lrfn(epoch):
    rates = [1e-3,1e-3,1e-4,1e-5,1e-6]
    return rates[epoch]
    
lr = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

rng = [i for i in range(5)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y,'-o'); 
plt.xlabel('epoch',size=14); plt.ylabel('learning rate',size=14)
plt.title('Training Schedule',size=16); 
#plt.show()

# %% [markdown]
# # Train Model

# %%
WEIGHT_PATH = './'

# %%
#for fold in range(5) Daha kısa sürede dönmesi için 1'e çektim
for fold in range(5):
    if fold>DO_FOLDS-1: continue
    
    print('#'*25)
    print('### FOLD %i'%(fold+1))
    
    # TRAIN DATA
    if TRAIN_WITH_TEST:
        train = raw.loc[ (raw.fold!=fold)&(raw.dcount>0)&
                    ((raw.istest==0)|( (raw.istest==1)&(raw.icount>0)&(raw.reverse==0) )) ].copy()
    else:
        train = raw.loc[ (raw.fold!=fold)&(raw.dcount>0)&(raw.istest==0) ].copy()
        
    # VALIDATION DATA
    valid = raw.loc[ (raw.fold==fold)&(raw.istest==0)&(raw.icount==0)&(raw.N>=4)&(raw.reverse==0) ].copy()

    print('### train shape',train.shape,'valid shape', valid.shape)    
    print('#'*25)
        
    # SAVE BEST VAL SCORE EPOCH MODEL
    sv = tf.keras.callbacks.ModelCheckpoint(
        f'{WEIGHT_PATH}/MLPx_fold{fold}_v{VER}.h5', monitor='val_weighted_sum_sparse_top_k_categorical_accuracy', verbose=1, 
        save_best_only=True, save_weights_only=True, mode='max', save_freq='epoch'
    )

    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, filename):
            super().__init__()
            self.filename = filename

        def on_epoch_end(self, epoch, logs=None):
            # Eğitim sırasında her epoch tamamlandığında çağrılır
            log_string = (
                f"###################################################\n"
                f"Epoch {epoch + 1}/{self.params['epochs']} - \n"
                f"Loss: {logs['loss']:.4f} - \n"
                #f"Main_output_loss: {logs['main_output_loss']:.4f} - \n"
                #f"emb_dot_soft_max_loss: {logs['emb_dot_soft_max_loss']:.4f} - \n"
                #f"weighted_sum_loss: {logs['weighted_sum_loss']:.4f}\n"
                #f"main_output_sparse_top_k_categorical_accuracy: {logs['main_output_sparse_top_k_categorical_accuracy']:.4f}\n"
                #f"emb_dot_soft_max_sparse_top_k_categorical_accuracy: {logs['emb_dot_soft_max_sparse_top_k_categorical_accuracy']:.4f}\n"
                #f"emb_dot_soft_max_1_sparse_top_k_categorical_accuracy: {logs['emb_dot_soft_max_1_sparse_top_k_categorical_accuracy']:.4f}\n"
                # f"weighted_sum_sparse_top_k_categorical_accuracy: {logs['weighted_sum_sparse_top_k_categorical_accuracy']:.4f}\n"
                # f"val_loss: {logs['val_loss']:.4f}\n"
                # f"val_main_output_loss: {logs['val_main_output_loss']:.4f}\n"
                # f"val_emb_dot_soft_max_loss: {logs['val_emb_dot_soft_max_loss']:.4f}\n"
                # f"val_emb_dot_soft_max_1_loss: {logs['val_emb_dot_soft_max_1_loss']:.4f}\n"
                # f"val_weighted_sum_loss: {logs['val_weighted_sum_loss']:.4f}\n"
                # f"val_main_output_sparse_top_k_categorical_accuracy: {logs['val_main_output_sparse_top_k_categorical_accuracy']:.4f}\n"
                # f"val_emb_dot_soft_max_sparse_top_k_categorical_accuracy: {logs['val_emb_dot_soft_max_sparse_top_k_categorical_accuracy']:.4f}\n"
                # f"val_emb_dot_soft_max_1_sparse_top_k_categorical_accuracy: {logs['val_emb_dot_soft_max_1_sparse_top_k_categorical_accuracy']:.4f}\n"
                f"val_weighted_sum_sparse_top_k_categorical_accuracy: {logs['val_weighted_sum_sparse_top_k_categorical_accuracy']:.4f}\n"
                f"###################################################\n"
            )

            # Başarı oranlarını dosyaya yaz
            with open(self.filename, 'a') as file:
                file.write(log_string)
            
    aum = CustomCallback(filename='accuracy_logs.txt')

    # wandb.init()
    # wandb.log({"Accuracy": (sv.monitor)})
    
    # GloVe embeddingleri yükle
    embedding_dim = 100  # GloVe embedding boyutu
    embedding_matrix = load_glove_embeddings(glove_embedding_path, word_index, embedding_dim)

    # GloVe embeddingleri kullanarak modeli oluştur
    model_with_glove = build_model_with_glove_embedding(embedding_matrix)

    model_with_glove.fit(train[FEATURES].to_pandas(),train[TARGET].to_pandas(),
          validation_data = (valid[FEATURES].to_pandas(),valid[TARGET].to_pandas()),
          epochs=5 #epochs=5
          ,verbose=1,batch_size=512, callbacks=[sv,lr,aum])
    
    del train, valid
    gc.collect()

# %% [markdown]
# # Validate Full OOF

# %%
city_mapping = cudf.DataFrame(MAPS[0]).reset_index()
#city_mapping = pd.Series(city_mapping.astype('int'), name='city_mapping') 
#city_mapping = MAPS[0].reset_index() Kodun ilk hali bu
#MAPS[0] = pd.DataFrame(MAPS)
#city_mapping = MAPS[0].reset_index(drop=True)

# %%
# PREDICT IN CHUNKS
# OTHERWISE DGX DIES

CHUNK = 1024*4

models = []
for k in range(DO_FOLDS):
    if k>DO_FOLDS-1: continue
    model = build_model()
    name = f'{WEIGHT_PATH}/MLPx_fold{k}_v{VER}.h5'
    print(name)
    model.load_weights(name)
    models.append(model)

valid = []

#for fold in range(5)
for fold in range(5):
    if fold>DO_FOLDS-1: continue
    print('#'*25)
    print('### FOLD %i'%(fold+1))
    
    test = raw.loc[ (raw.N>=4)&(raw.fold==fold)&(raw.reverse==0)&
                   ( ((raw.istest==0)&(raw.icount<=1)) | ((raw.istest==1)&(raw.icount==1)) ) ].copy()
        
    print('### valid shape', test.shape )
    print('#'*25)

    test.reset_index(drop=True,inplace=True)
    TOP4 = np.zeros((test.shape[0],4))

    print( test.shape )
    for k in range(test.shape[0]//CHUNK + 1):
        a = k*CHUNK
        b = (k+1)*CHUNK
        b = min(test.shape[0],b)
    
        print('Fold %i Chunk %i to %i'%(fold,a,b))
        _,_,_,preds = models[fold].predict(test[FEATURES].iloc[a:b].to_pandas(),verbose=0,batch_size=512)
        for i in range(4):
            x = np.argmax(preds,axis=1)
            TOP4[a:b,i] = x
            for j in range(preds.shape[0]):
                preds[j,x[j]] = -1
                
    for k in range(4):
        test['rec%i'%k] = TOP4[:,k] - 1
        tmp = test[['rec%i'%k]].astype('int32').copy()
        tmp['sorting'] = cupy.asarray(range(tmp.shape[0]))
        tmp = tmp.merge(city_mapping, how='left', left_on='rec%i'%k, right_on='index')
        tmp = tmp.sort_values('sorting')
        test['rec%i'%k] = tmp['rec%i'%k].values.copy()
        #test['rec%i'%k] = tmp['city_id'].values.copy()
    valid.append(test)

# %%
if DO_FOLDS==1:
    valid = valid[0]
else:
    valid = cudf.concat(valid,axis=0,ignore_index=True)

# %%
# VALIDATION LAST CITY - FULL OOF
top4_metric( valid, 0, 0, target='city_id' )

# %%
# VALIDATION 2ND LAST CITY - FULL OOF
top4_metric( valid, 0, 1, target='city_id' )

# %%
# TEST 2ND LAST CITY 
top4_metric( valid, 1, 1, target='city_id' )

# %% [markdown]
# # Validate Test 2nd Last City - 5 Folds

# %%
test = raw.loc[ (raw.istest==1)&(raw.icount==1)&(raw.reverse==0) ].copy()
print( test.shape )
test.head()

# %%
# PREDICT IN CHUNKS
# OTHERWISE DGX DIES

CHUNK = 1024*8
test.reset_index(drop=True,inplace=True)
TOP4 = np.zeros((test.shape[0],4))

print( test.shape )
for k in range(test.shape[0]//CHUNK + 1):
    if (not DO_TEST)|(TRAIN_WITH_TEST): continue
    a = k*CHUNK
    b = (k+1)*CHUNK
    b = min(test.shape[0],b)
    
    preds = np.zeros((b-a,t_ct))
    for j in range(1):
        print('Fold %i Chunk %i to %i'%(j,a,b))
        test_ = test[FEATURES].iloc[a:b].copy()      
        _,_,_,preds0 = models[j].predict(test_.to_pandas(),verbose=0,batch_size=512)
        preds += preds0
    preds /= 1.0
        
    for i in range(4):
        x = np.argmax(preds,axis=1)
        TOP4[a:b,i] = x
        for j in range(preds.shape[0]):
            preds[j,x[j]] = -1

# %%
if not((not DO_TEST)|(TRAIN_WITH_TEST)):
    for k in range(4):
        test['rec%i'%k] = TOP4[:,k] - 1
        tmp = test[['rec%i'%k]].astype('int32').copy()
        tmp['sorting'] = cupy.asarray(range(tmp.shape[0]))
        tmp = tmp.merge(city_mapping, how='left', left_on='rec%i'%k, right_on='index')
        tmp = tmp.sort_values('sorting')
        test['rec%i'%k] = tmp['rec%i'%k].values.copy()
        #test['rec%i'%k] = tmp['city_id'].values.copy()
    # TEST 2ND LAST CITY
    # WITHOUT FOLD ENSEMBLE HAS 0.4278
    top4_metric( test, 1, 1, target='city_id' )

# %% [markdown]
# # Predict Test Last City Data - 5 Folds - Submission.csv

# %%
test = raw.loc[ (raw.istest==1)&(raw.icount==0)&(raw.reverse==0) ].copy()
print( test.shape )
test.head()

# %%
# PREDICT IN CHUNKS
# OTHERWISE DGX DIES
CHUNK = 1024*8
test.reset_index(drop=True,inplace=True)

TOP4 = np.zeros((test.shape[0],4))

print( test.shape )
for k in range(test.shape[0]//CHUNK + 1):
    if not DO_TEST: continue
    a = k*CHUNK
    b = (k+1)*CHUNK
    b = min(test.shape[0],b)
        
    preds = np.zeros((b-a,t_ct))
    for j in range(DO_FOLDS):
        print('Fold %i Chunk %i to %i'%(j,a,b))
        test_ = test[FEATURES].iloc[a:b].copy()       
        _,_,_,preds0 = models[j].predict(test_.to_pandas(),verbose=0,batch_size=512)
        preds += preds0
    preds /= 5.0
    
    for i in range(4):
        x = np.argmax(preds,axis=1)
        TOP4[a:b,i] = x
        for j in range(preds.shape[0]):
            preds[j,x[j]] = -1

# %%
COLS = ['utrip_id']
for k in range(4):
    test['city_id_%i'%(k+1)] = TOP4[:,k] - 1
    tmp = test[['city_id_%i'%(k+1)]].astype('int32').copy()
    tmp['sorting'] = cupy.asarray(range(tmp.shape[0]))
    tmp = tmp.merge(city_mapping, how='left', left_on='city_id_%i'%(k+1), right_on='index')
    tmp = tmp.sort_values('sorting')
    test['city_id_%i'%(k+1)] = tmp['city_id_%i'%(k+1)].values.copy()
    COLS.append('city_id_%i'%(k+1))

# %%
test[COLS].head()

# %%
if DO_TEST:
    test[COLS].to_csv('submission-MLPx-RNN_v%i.csv'%VER,index=False)

print("BAŞARDINIZ") 
#wandb.finish()


=======
# %% [markdown]
# The MIT License (MIT)
# 
# Copyright (c) 2021 NVIDIA CORPORATION
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# %% [markdown]
# # GRU with MultiStage Session-based Matrix Factorization head

# %%
import os, sys

VER = 44
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Sistemde hangi gpu'nun çalışacağını belirler
MAX_GPU_MEMORY_MB = 20000 #hafızayı aşmaması için ekledim 0410


TRAIN_WITH_TEST = True
# ONLY DO THIS MANY FOLDS
#DO_FOLDS = 5 Daha kısa sürede dönmesi için 1'e çektim
DO_FOLDS = 5
# MAKE SUBMISSION OR NOT
DO_TEST = True

# %%
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import cudf
import cupy
import gc
from sklearn.model_selection import GroupKFold
import wandb
import random
import keyboard
import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, GRU
# from tensorflow.keras.models import Model
pd.set_option('display.max_columns', None)


pd.__version__, cudf.__version__

# %% [markdown]
# # Metric Calculation

# %%
# Return top4 metric
# istest: flag to select if metric should be computed in 0:train, 1:test, -1:both
# pos: select which city to calculate the metric, 0: last, 1: last-1, 2:last-2 , -1: all
# the input `val` dataframe must contains the target `city_id` and the 4 recommendations as: rec0, res1, rec2 and rec3
    
def top4_metric( val, istest=0, pos=0 , target='city_id'):
    
    if istest>=0:
        val = val.loc[ (val.submission==0) & (val.istest == istest) ]
    else:
        val = val.loc[ (val.submission==0) ]

    if pos >= 0:
        top1 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec0']
        top2 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec1']
        top3 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec2']
        top4 = val.loc[val.icount==pos,target] == val.loc[val.icount==pos,'rec3']
    else:
        top1 = val[target] == val['rec0']
        top2 = val[target] == val['rec1']
        top3 = val[target] == val['rec2']
        top4 = val[target] == val['rec3']
        
    return (top1|top2|top3|top4).mean()  

##Wandb Uygulaması
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="GRU-MS-SMF-LessData",
#     config= {
#             "TRAIN_BATCH_SIZE" : 512,
#             "WORKERS" : 8,
#             "LR" : 1e-3,
#             "EPOCHS" : 5,
#             "GRADIENT_ACCUMULATION" : 1,
#             "EMBEDDING_DIM" : 64,
#             "HIDDEN_DIM" :  1024,
#             "DROPOUT_RATE" : 0.4,
#             }
# )  

# %% [markdown]
# # Feature Engineering

# %%
#%%time
PATH = './'
#raw = cudf.read_csv('../../00_Data/train_and_test_2.csv')
raw = cudf.read_csv('00_Data/train_and_test_2.csv')
print(raw.shape)

# %%
raw['reverse'] = 0
raw.head()

# %%
reverse = raw.copy()
reverse['reverse'] = 1
reverse['utrip_id'] = reverse['utrip_id']+'_r'

# %%
reverse['icount'] = reverse['dcount'].copy()

# %%
reverse = reverse.sort_values(['utrip_id', 'icount'], ascending=False)
raw = cudf.concat([raw, reverse], axis=0)
raw['sorting'] = cupy.asarray(range(raw.shape[0]))
del reverse
gc.collect()

# %%
raw['utrip_id'+'_'], mp = raw['utrip_id'].factorize()

# %%
#%%time

raw['checkin'] = cudf.to_datetime(raw.checkin, format="%Y-%m-%d")
raw['checkout'] = cudf.to_datetime(raw.checkout, format="%Y-%m-%d")

# %%
#%%time

# ENGINEER MONTH, WEEKDAY, and LENGTH OF STAY FEATURES
raw['mn'] = raw.checkin.dt.month
raw['dy1'] = raw.checkin.dt.weekday
raw['dy2'] = raw.checkout.dt.weekday
raw['length'] = (raw.checkout - raw.checkin).dt.days

# ENGINEER WEEKEND AND SEASON
raw['day_name']= raw.checkin.dt.weekday
raw['isweekend']=raw['day_name'].isin([5,6]).astype('int8')


# %%
df_season = cudf.DataFrame({'mn': range(1,13), 'season': ([0]*3)+([1]*3)+([2]*3)+([3]*3)})

# %%
raw=raw.merge(df_season, how='left', on='mn')
raw = raw.sort_values(['sorting'], ascending=True)

# %%
raw.head()

# %%
def shift_feature(df, groupby_col, col, offset, nan=-1, colname=''):
    df[colname] = df[col].shift(offset)
    df.loc[df[groupby_col]!=df[groupby_col].shift(offset), colname] = nan

# %%
# WEIRD FEATURE IS DIFFERENT FOR REVERSE AND NORMAL
shift_feature(raw, 'utrip_id_', 'checkout', 1, None, 'checkout_lag1')
raw['gap'] = (raw.checkin - raw.checkout_lag1).dt.days.fillna(-1) 

# %%
#%%time

# REVERSE ICOUNT
raw['dcount'] = raw['N']-raw['icount']-1

# %%
raw.head()

# %%
# LABEL ENCODE CATEGORICAL
CATS = ['city_id','mn','dy1','dy2','length','device_class','affiliate_id','booker_country',
        'hotel_country','icount','dcount','gap','isweekend', 'season']
MAPS = []
for c in CATS:
    raw[c+'_'], mp = raw[c].factorize()
    MAPS.append(mp)
    print('created',c+'_')

# %%
LAGS = 5
EC = 400

# ENGINEER LAG FEATURES
for i in range(1,LAGS+1):
    shift_feature(raw, 'utrip_id_', 'city_id_', i, -1, f'city_id_lag{i}')

# %%
shift_feature(raw, 'utrip_id_', 'hotel_country_', i, -1, 'hotel_lag1')

# %%
# TOTAL DURATION AND FIRST CITY
tmpA = raw[raw['icount']==0][['utrip_id', 'checkout']]
tmpB = raw[raw['dcount']==0][['utrip_id', 'checkin']]
tmpC = tmpA.merge(tmpB, how='left', on='utrip_id')
tmpC['length_total'] = (tmpC['checkout'] - tmpC['checkin']).dt.days
raw = raw.merge(tmpC[['utrip_id', 'length_total']],on='utrip_id',how='left')

# %%
tmpD = raw[raw['dcount']==0][['utrip_id', 'city_id_']]
tmpD.columns = ['utrip_id', 'city_first']
raw = raw.merge(tmpD,on='utrip_id',how='left')

# %%
tmpA = raw[raw['dcount']==0][['utrip_id', 'checkout']]
tmpB = raw[raw['icount']==0][['utrip_id', 'checkin']]
tmpC = tmpA.merge(tmpB, how='left', on='utrip_id')
tmpC['length_total_r'] = (tmpC['checkout'] - tmpC['checkin']).dt.days
raw = raw.merge(tmpC[['utrip_id', 'length_total_r']],on='utrip_id',how='left')
raw.loc[raw['length_total']<0,'length_total'] = raw.loc[raw['length_total']<0,'length_total_r']

# %%
raw[raw['utrip_id'].isin(['999944_1','999944_1_r'])]

# %% [markdown]
# # Finish Feature Engineering

# %%
# CONVERT -1 TO 0
cols = [f'city_id_lag{i}' for i in range(LAGS,0,-1)]
for c in cols+['city_id_','city_first']+['hotel_lag1']:
    raw[c] += 1
    print(c)

# %%
# MOVE RARE CITIES INTO COMMON CLASS
RARE = 9*2 #x2 because data has doubled
tmp = raw.city_id_.value_counts()
idx = tmp.loc[tmp<=RARE].index.values
print('rare city ct under <=%i'%RARE, len(idx),'total cities =',len(tmp) )

mx = raw.city_id_.max()+1
for c in cols+['city_first']:
    raw.loc[raw[c].isin(idx),c] = mx

# %%
# VERIFY LAG LOOKS RIGHT
raw[['city_id_','city_id_','icount']+cols].head()

# %%
FEATURES = ['mn_', 'dy1_','dy2_','length_','device_class_',  'affiliate_id_', 'booker_country_',
       'hotel_lag1','icount','dcount','city_first','length_total','gap_','isweekend_', 'season_'] + cols
TARGET = ['city_id_']

# %% [markdown]
# **Note that the order should be ['city_id_lag3', 'city_id_lag2', 'city_id_lag1'] in the order of time for the sake of RNN**

# %%
cols

# %% [markdown]
# # GRU-MS-SMF 5 Fold Model

# %%
os.environ['TF_MEMORY_ALLOCATION'] = "0.7" # fraction of free memory

# %%
import tensorflow as tf
tf.__version__

# %%
gpus = tf.config.list_physical_devices('GPU')
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7) #hafıza hatası almamak için ekledim 0410
# %%
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*19)]
)

# %% [markdown]
# # Define Embeddings

# %%
# EMBEDDING INPUT SIZES
cts = []
for c in FEATURES:
    mx = raw[c].max()
    cts.append(mx+1)
    print(c,mx+1)

# %%
# EMBEDDING OUTPUT SIZES
#EC = 400
# Özelliklerin embedding boyutlarını belirleyin
EMBEDDING_DIMS = [4, 3, 3, 6, 2, 60, 3, 14, 7, 7, 200, 18, 9, 3, 4, 400, 400, 400, 400, 400]

cts2 = [4,3,3,6,2,60,3,14,7,7,200,18,9,3,4]+[EC]*len(cols)

# %%
emb_map = {i:(j,k) for i,j,k in zip(FEATURES, cts, cts2)}
emb_map

# %%
# TARGET SOFTMAX OUTPUT SIZE
t_ct = raw['city_id_'].max()+1
t_ct

# %% [markdown]
# # Build Model

# %%
N_CITY = 50

class Linear(tf.keras.layers.Layer):
    def __init__(self, H, activation='relu'):
        super(Linear, self).__init__()
        self.dense = tf.keras.layers.Dense(H)
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class WeightedSum(tf.keras.layers.Layer):
    def __init__(self):
        super(WeightedSum, self).__init__(name='weighted_sum')

    def build(self, input_shape):
        self.w1 = self.add_weight(name='w1',shape=None,dtype=tf.float32,
                                 trainable=True)
        self.w2 = self.add_weight(name='w2',shape=None,dtype=tf.float32,
                                 trainable=True)
        
    def call(self, x1, x2, x3):
        w1 = tf.nn.sigmoid(self.w1)
        w2 = tf.nn.sigmoid(self.w2)
        x1 = tf.stop_gradient(x1)
        x2 = tf.stop_gradient(x2)
        x3 = tf.stop_gradient(x3)
        return (x1 + x2*w1 + x3*w2)/(1+w1+w2)
    
class EmbDotSoftMax(tf.keras.layers.Layer):
    def __init__(self):
        super(EmbDotSoftMax, self).__init__()
        self.d1 = tf.keras.layers.Dense(EC) 
    
    def call(self, x, top_city_emb, top_city_id, prob):
        emb_pred = self.d1(x) # B,EC
        emb_pred = tf.expand_dims(emb_pred, axis=1) #B,1,EC
        x = emb_pred*top_city_emb #B,N_CITY,EC
        x = tf.math.reduce_sum(x, axis=2) #B,N_CITY
        x = tf.nn.softmax(x) #B,N_CITY

        rowids = tf.range(0,tf.shape(x)[0]) # B
        rowids = tf.transpose(tf.tile([rowids],[N_CITY,1])) # B,N_CITY

        idx = tf.stack([rowids,top_city_id],axis=2) # B, N_CITY, 2
        idx = tf.cast(idx, tf.int32)
        prob = tf.scatter_nd(idx, x, tf.shape(prob)) + 1e-6
        return prob
    
glove_embedding_path = '00_Data/glove.6B.100d.txt'  # Örneğin, dosya yolu

# Veri setinizdeki metinsel özellikler için bir kelime dizini oluşturun
word_index = {"The Devilfire Empire":1,
"Mundania":2,
"Takistan":3,
"Nevoruss":4,
"Sarkhan":5,
"Dawsbergen":6,
"Genosha":7,
"Saint Marie":8,
"West Hun Chiu":9,
"Ragaan":10,
"Squamuglia":11,
"Pullamawang":12,
"Svenborgia":13,
"Aslerfan":14,
"Yellow Empire":15,
"Slovetzia":16,
"Nambutu":17,
"Gilead":18,
"Florin":19,
"Novoselic":20,
"Phaic TÄƒn":21,
"Matobo":22,
"Syldavia":23,
"Utopia":24,
"Turgistan":25,
"Costa Luna":26,
"Veyshnoria":27,
"Feldenberg":28,
"Altis and Stratis":29,
"Kumbolaland":30,
"Sokovia":31,
"Bartovia":32,
"Tcherkistan":33,
"Durhan":34,
"Osterlich":35,
"Guilder":36,
"Wadiya":37,
"Maltovia":38,
"Merania":39,
"Penguina (L'Ã®le des Pingouins)":40,
"Shtischtorchnia":41,
"Grinlandia":42,
"Zephyria":43,
"Wredpryd":44,
"Vulgaria":45,
"Fook Island":46,
"Moronika":47,
"Datlof":48,
"Vadeem":49,
"San Marcos":50,
"Bultan":51,
"Franchia":52,
"Laurania":53,
"Leutonia":54,
"Kazahrus":55,
"Kyrat":56,
"Pokolistan":57,
"Panem":58,
"Sahrani":59,
"Oceania":60,
"West Angola":61,
"Bacteria":62,
"Sardovia":63,
"Markovia":64,
"Borginia":65,
"Outer Heaven":66,
"New Germany":67,
"Eurasia":68,
"Polrugaria":69,
"Aldovia":70,
"Kunami":71,
"Grenyarnia":72,
"Elbonia":73,
"Kangan":74,
"Tyranistan":75,
"Alvonia":76,
"Cobra Island":77,
"Basran":78,
"Pokrovia":79,
"Bruzundanga":80,
"Tsergovia":81,
"Samavia":82,
"Chinese Federation":83,
"Urk (also Uruk)":84,
"Latveria":85,
"Kamistan":86,
"Romanza":87,
"Idris":88,
"Atlantis":89,
"GÃ©rolstein":90,
"Angrezi Raj":91,
"Nairomi":92,
"Coalition States":93,
"Medici":94,
"":95,
"Glubbdubdrib":96,
"Buranda":97,
"San Theodoros":98,
"Sylvania":99,
"Zekistan":100,
"Brobdingnag":101,
"Marina Venetta":102,
"Lugash":103,
"Carpathia":104,
"Nerdocrumbesia":105,
"Shangri-La":106,
"Gondour":107,
"Danu":108,
"Congaree Socialist Republic":109,
"Holy Britannian Empire":110,
"Baltish":111,
"Trans-Carpathia":112,
"Axphain":113,
"Patusan":114,
"Almaigne":115,
"Norteguay":116,
"Groland":117,
"Urmania":118,
"Carjackistan":119,
"Isle of Fogg":120,
"Kahndaq":121,
"San SombrÃ¨ro":122,
"Aldorria":123,
"Lower Slobbovia":124,
"Tirania":125,
"Genovia":126,
"Flausenthurm":127,
"Slaka":128,
"Republic of New Rearendia":129,
"MolvanÃ®a":130,
"Freedonia":131,
"Novistrana":132,
"Buenaventura":133,
"Santa Prisca":134,
"Rhelasia":135,
"Yudonia":136,
"Uqbar":137,
"Robo-Hungarian Empire":138,
"Braganza":139,
"Qasha":140,
"Nuevo Rico":141,
"Rook Islands":142,
"Palombia":143,
"Russian Democratic Union":144,
"Bandaria":145,
"Kazirstan":146,
"Metrofulus":147,
"Absurdistan":148,
"North American Union":149,
"Poictesme":150,
"Halla":151,
"Sodor":152,
"Kasnia":153,
"El Othar":154,
"Bialya":155,
"Tijata":156,
"Marshovia":157,
"IllÃ©a":158,
"Lilliput":159,
"Edonia":160,
"St. George's Island":161,
"Graustark":162,
"Drusselstein":163,
"Babar's Kingdom":164,
"Grand Fenwick":165,
"Graznavia":166,
"San Salvador":167,
"Krakozhia":168,
"Bolumbia":169,
"Gondal":170,
"Erewhon":171,
"San Serriffe":172,
"Yerba":173,
"Glovania":174,
"Taronia":175,
"Bahari":176,
"Moldavia":177,
"Chernarus":178,
"Urkesh":179,
"Khura'in":180,
"Bangalla":181,
"SÃ£o Rico":182,
"Norland":183,
"Caledonia":184,
"Bahavia":185,
"Naruba":186,
"Mypos":187,
"Sunda":188,
"Illyria":189,
"San Lorenzo":190,
"Lovitzna":191,
"Rolisica":192,
"Bozatta":193,
"Isla Island":194,
"Nova Africa":195,
"Borostyria":196}  # Örnek olarak, gerçek verilerle doldurulmalıdır.

def load_glove_embeddings(glove_embedding_path, word_index, embedding_dim):

    # Load pre-trained GloVe embeddings
    print("Loading GloVe embeddings...")
    embeddings_index = {}
    with open(glove_embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("GloVe embeddings loaded successfully.")

    # Create an embedding matrix
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# GloVe embeddinglerini yükleme fonksiyonu
def build_model_with_glove_embedding(embedding_matrix):
    inp = tf.keras.layers.Input(shape=(len(FEATURES),))
    embs = []
    
    for k, f in enumerate(FEATURES):
        if f in ["booker_country", "hotel_country"]:
            # Metinsel özellikler için GloVe embeddinglerini kullanın
            embedding_dim = EMBEDDING_DIMS[k]  # EMBEDDING_DIMS listesinden boyutu alın
            num_words = len(word_index) + 1  # Kelime dizinindeki kelime sayısı + 1
            emb_layer = tf.keras.layers.Embedding(num_words, embedding_dim, weights=[embedding_matrix], trainable=False)
            emb = emb_layer(inp[:, k])
            emb = tf.keras.layers.Reshape((1, embedding_dim))(emb)  # Embedding boyutunu 1xembedding_dim yapın
            embs.append(emb)
        else:
            # Diğer özellikler için normal embedding katmanlarını kullanın
            i, j = emb_map[f]
            embedding_dim = EMBEDDING_DIMS[k]  # EMBEDDING_DIMS listesinden boyutu alın
            emb = tf.keras.layers.Embedding(i, embedding_dim)(inp[:, k])
            emb = tf.keras.layers.Reshape((1, embedding_dim))(emb)  # Embedding boyutunu 1xembedding_dim yapın
            embs.append(emb)
    
    # Tüm girişleri aynı boyuta getirin
    x = tf.keras.layers.Concatenate(axis=1)(embs)
    
    xc = tf.keras.layers.GRU(128, activation='tanh')(x)
    
    x1 = tf.keras.layers.Dense(512+256, activation='relu')(x)
    x2 = tf.keras.layers.Dense(512+256, activation='relu')(x1)
    prob = tf.keras.layers.Dense(t_ct, activation='softmax', name='main_output')(x2)
    
    _, top_city_id = tf.math.top_k(prob, N_CITY)
    top_city_emb = emb_map['city_id_lag1'](top_city_id)
    
    x1 = tf.keras.layers.Dense(512+256, activation='relu')(x1)
    prob_1 = EmbDotSoftMax()(x1, top_city_emb, top_city_id, prob)
    prob_2 = EmbDotSoftMax()(x2, top_city_emb, top_city_id, prob)
    
    prob_ws = WeightedSum()(prob, prob_1, prob_2)
    
    model = tf.keras.models.Model(inputs=inp, outputs=[prob, prob_1, prob_2, prob_ws])
    opt = tf.keras.optimizers.Adam(lr=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    mtr = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=4)
    model.compile(loss=loss, optimizer=opt, metrics=[mtr])
    return model

# %% [markdown]
# # Learning Schedule

# %%
# CUSTOM LEARNING SCHEUDLE

def lrfn(epoch):
    rates = [1e-3,1e-3,1e-4,1e-5,1e-6]
    return rates[epoch]
    
lr = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

rng = [i for i in range(5)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y,'-o'); 
plt.xlabel('epoch',size=14); plt.ylabel('learning rate',size=14)
plt.title('Training Schedule',size=16); 
#plt.show()

# %% [markdown]
# # Train Model

# %%
WEIGHT_PATH = './'

# %%
#for fold in range(5) Daha kısa sürede dönmesi için 1'e çektim
for fold in range(5):
    if fold>DO_FOLDS-1: continue
    
    print('#'*25)
    print('### FOLD %i'%(fold+1))
    
    # TRAIN DATA
    if TRAIN_WITH_TEST:
        train = raw.loc[ (raw.fold!=fold)&(raw.dcount>0)&
                    ((raw.istest==0)|( (raw.istest==1)&(raw.icount>0)&(raw.reverse==0) )) ].copy()
    else:
        train = raw.loc[ (raw.fold!=fold)&(raw.dcount>0)&(raw.istest==0) ].copy()
        
    # VALIDATION DATA
    valid = raw.loc[ (raw.fold==fold)&(raw.istest==0)&(raw.icount==0)&(raw.N>=4)&(raw.reverse==0) ].copy()

    print('### train shape',train.shape,'valid shape', valid.shape)    
    print('#'*25)
        
    # SAVE BEST VAL SCORE EPOCH MODEL
    sv = tf.keras.callbacks.ModelCheckpoint(
        f'{WEIGHT_PATH}/MLPx_fold{fold}_v{VER}.h5', monitor='val_weighted_sum_sparse_top_k_categorical_accuracy', verbose=1, 
        save_best_only=True, save_weights_only=True, mode='max', save_freq='epoch'
    )

    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self, filename):
            super().__init__()
            self.filename = filename

        def on_epoch_end(self, epoch, logs=None):
            # Eğitim sırasında her epoch tamamlandığında çağrılır
            log_string = (
                f"###################################################\n"
                f"Epoch {epoch + 1}/{self.params['epochs']} - \n"
                f"Loss: {logs['loss']:.4f} - \n"
                #f"Main_output_loss: {logs['main_output_loss']:.4f} - \n"
                #f"emb_dot_soft_max_loss: {logs['emb_dot_soft_max_loss']:.4f} - \n"
                #f"weighted_sum_loss: {logs['weighted_sum_loss']:.4f}\n"
                #f"main_output_sparse_top_k_categorical_accuracy: {logs['main_output_sparse_top_k_categorical_accuracy']:.4f}\n"
                #f"emb_dot_soft_max_sparse_top_k_categorical_accuracy: {logs['emb_dot_soft_max_sparse_top_k_categorical_accuracy']:.4f}\n"
                #f"emb_dot_soft_max_1_sparse_top_k_categorical_accuracy: {logs['emb_dot_soft_max_1_sparse_top_k_categorical_accuracy']:.4f}\n"
                # f"weighted_sum_sparse_top_k_categorical_accuracy: {logs['weighted_sum_sparse_top_k_categorical_accuracy']:.4f}\n"
                # f"val_loss: {logs['val_loss']:.4f}\n"
                # f"val_main_output_loss: {logs['val_main_output_loss']:.4f}\n"
                # f"val_emb_dot_soft_max_loss: {logs['val_emb_dot_soft_max_loss']:.4f}\n"
                # f"val_emb_dot_soft_max_1_loss: {logs['val_emb_dot_soft_max_1_loss']:.4f}\n"
                # f"val_weighted_sum_loss: {logs['val_weighted_sum_loss']:.4f}\n"
                # f"val_main_output_sparse_top_k_categorical_accuracy: {logs['val_main_output_sparse_top_k_categorical_accuracy']:.4f}\n"
                # f"val_emb_dot_soft_max_sparse_top_k_categorical_accuracy: {logs['val_emb_dot_soft_max_sparse_top_k_categorical_accuracy']:.4f}\n"
                # f"val_emb_dot_soft_max_1_sparse_top_k_categorical_accuracy: {logs['val_emb_dot_soft_max_1_sparse_top_k_categorical_accuracy']:.4f}\n"
                f"val_weighted_sum_sparse_top_k_categorical_accuracy: {logs['val_weighted_sum_sparse_top_k_categorical_accuracy']:.4f}\n"
                f"###################################################\n"
            )

            # Başarı oranlarını dosyaya yaz
            with open(self.filename, 'a') as file:
                file.write(log_string)
            
    aum = CustomCallback(filename='accuracy_logs_2404_GloVe.txt')

    # wandb.init()
    # wandb.log({"Accuracy": (sv.monitor)})
    
    # GloVe embeddingleri yükle
    embedding_dim = 100  # GloVe embedding boyutu
    embedding_matrix = load_glove_embeddings(glove_embedding_path, word_index, embedding_dim)

    # GloVe embeddingleri kullanarak modeli oluştur
    model_with_glove = build_model_with_glove_embedding(embedding_matrix)

    model_with_glove.fit(train[FEATURES].to_pandas(),train[TARGET].to_pandas(),
          validation_data = (valid[FEATURES].to_pandas(),valid[TARGET].to_pandas()),
          epochs=5 #epochs=5
          ,verbose=1,batch_size=512, callbacks=[sv,lr,aum])
    
    del train, valid
    gc.collect()

# %% [markdown]
# # Validate Full OOF

# %%
city_mapping = cudf.DataFrame(MAPS[0]).reset_index()
#city_mapping = pd.Series(city_mapping.astype('int'), name='city_mapping') 
#city_mapping = MAPS[0].reset_index() Kodun ilk hali bu
#MAPS[0] = pd.DataFrame(MAPS)
#city_mapping = MAPS[0].reset_index(drop=True)

# %%
# PREDICT IN CHUNKS
# OTHERWISE DGX DIES

CHUNK = 1024*4

models = []
for k in range(DO_FOLDS):
    if k>DO_FOLDS-1: continue
    model = build_model_with_glove_embedding(embedding_matrix)
    name = f'{WEIGHT_PATH}/MLPx_fold{k}_v{VER}.h5'
    print(name)
    model.load_weights(name)
    models.append(model)

valid = []

#for fold in range(5)
for fold in range(5):
    if fold>DO_FOLDS-1: continue
    print('#'*25)
    print('### FOLD %i'%(fold+1))
    
    test = raw.loc[ (raw.N>=4)&(raw.fold==fold)&(raw.reverse==0)&
                   ( ((raw.istest==0)&(raw.icount<=1)) | ((raw.istest==1)&(raw.icount==1)) ) ].copy()
        
    print('### valid shape', test.shape )
    print('#'*25)

    test.reset_index(drop=True,inplace=True)
    TOP4 = np.zeros((test.shape[0],4))

    print( test.shape )
    for k in range(test.shape[0]//CHUNK + 1):
        a = k*CHUNK
        b = (k+1)*CHUNK
        b = min(test.shape[0],b)
    
        print('Fold %i Chunk %i to %i'%(fold,a,b))
        _,_,_,preds = models[fold].predict(test[FEATURES].iloc[a:b].to_pandas(),verbose=0,batch_size=512)
        for i in range(4):
            x = np.argmax(preds,axis=1)
            TOP4[a:b,i] = x
            for j in range(preds.shape[0]):
                preds[j,x[j]] = -1
                
    for k in range(4):
        test['rec%i'%k] = TOP4[:,k] - 1
        tmp = test[['rec%i'%k]].astype('int32').copy()
        tmp['sorting'] = cupy.asarray(range(tmp.shape[0]))
        tmp = tmp.merge(city_mapping, how='left', left_on='rec%i'%k, right_on='index')
        tmp = tmp.sort_values('sorting')
        test['rec%i'%k] = tmp['rec%i'%k].values.copy()
        #test['rec%i'%k] = tmp['city_id'].values.copy()
    valid.append(test)

# %%
if DO_FOLDS==1:
    valid = valid[0]
else:
    valid = cudf.concat(valid,axis=0,ignore_index=True)

# %%
# VALIDATION LAST CITY - FULL OOF
top4_metric( valid, 0, 0, target='city_id' )

# %%
# VALIDATION 2ND LAST CITY - FULL OOF
top4_metric( valid, 0, 1, target='city_id' )

# %%
# TEST 2ND LAST CITY 
top4_metric( valid, 1, 1, target='city_id' )

# %% [markdown]
# # Validate Test 2nd Last City - 5 Folds

# %%
test = raw.loc[ (raw.istest==1)&(raw.icount==1)&(raw.reverse==0) ].copy()
print( test.shape )
test.head()

# %%
# PREDICT IN CHUNKS
# OTHERWISE DGX DIES

CHUNK = 1024*8
test.reset_index(drop=True,inplace=True)
TOP4 = np.zeros((test.shape[0],4))

print( test.shape )
for k in range(test.shape[0]//CHUNK + 1):
    if (not DO_TEST)|(TRAIN_WITH_TEST): continue
    a = k*CHUNK
    b = (k+1)*CHUNK
    b = min(test.shape[0],b)
    
    preds = np.zeros((b-a,t_ct))
    for j in range(1):
        print('Fold %i Chunk %i to %i'%(j,a,b))
        test_ = test[FEATURES].iloc[a:b].copy()      
        _,_,_,preds0 = models[j].predict(test_.to_pandas(),verbose=0,batch_size=512)
        preds += preds0
    preds /= 1.0
        
    for i in range(4):
        x = np.argmax(preds,axis=1)
        TOP4[a:b,i] = x
        for j in range(preds.shape[0]):
            preds[j,x[j]] = -1

# %%
if not((not DO_TEST)|(TRAIN_WITH_TEST)):
    for k in range(4):
        test['rec%i'%k] = TOP4[:,k] - 1
        tmp = test[['rec%i'%k]].astype('int32').copy()
        tmp['sorting'] = cupy.asarray(range(tmp.shape[0]))
        tmp = tmp.merge(city_mapping, how='left', left_on='rec%i'%k, right_on='index')
        tmp = tmp.sort_values('sorting')
        test['rec%i'%k] = tmp['rec%i'%k].values.copy()
        #test['rec%i'%k] = tmp['city_id'].values.copy()
    # TEST 2ND LAST CITY
    # WITHOUT FOLD ENSEMBLE HAS 0.4278
    top4_metric( test, 1, 1, target='city_id' )

# %% [markdown]
# # Predict Test Last City Data - 5 Folds - Submission.csv

# %%
test = raw.loc[ (raw.istest==1)&(raw.icount==0)&(raw.reverse==0) ].copy()
print( test.shape )
test.head()

# %%
# PREDICT IN CHUNKS
# OTHERWISE DGX DIES
CHUNK = 1024*8
test.reset_index(drop=True,inplace=True)

TOP4 = np.zeros((test.shape[0],4))

print( test.shape )
for k in range(test.shape[0]//CHUNK + 1):
    if not DO_TEST: continue
    a = k*CHUNK
    b = (k+1)*CHUNK
    b = min(test.shape[0],b)
        
    preds = np.zeros((b-a,t_ct))
    for j in range(DO_FOLDS):
        print('Fold %i Chunk %i to %i'%(j,a,b))
        test_ = test[FEATURES].iloc[a:b].copy()       
        _,_,_,preds0 = models[j].predict(test_.to_pandas(),verbose=0,batch_size=512)
        preds += preds0
    preds /= 5.0
    
    for i in range(4):
        x = np.argmax(preds,axis=1)
        TOP4[a:b,i] = x
        for j in range(preds.shape[0]):
            preds[j,x[j]] = -1

# %%
COLS = ['utrip_id']
for k in range(4):
    test['city_id_%i'%(k+1)] = TOP4[:,k] - 1
    tmp = test[['city_id_%i'%(k+1)]].astype('int32').copy()
    tmp['sorting'] = cupy.asarray(range(tmp.shape[0]))
    tmp = tmp.merge(city_mapping, how='left', left_on='city_id_%i'%(k+1), right_on='index')
    tmp = tmp.sort_values('sorting')
    test['city_id_%i'%(k+1)] = tmp['city_id_%i'%(k+1)].values.copy()
    COLS.append('city_id_%i'%(k+1))

# %%
test[COLS].head()

# %%
if DO_TEST:
    test[COLS].to_csv('submission-MLPx-RNN_v%i.csv'%VER,index=False)

print("BAŞARDINIZ") 
#wandb.finish()


>>>>>>> c98dbca820bcfd102f1c062ba52aef178a7660f3
