import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import keras
import sklearn.model_selection as model_selection
from data_pipeline import *
from My_Models import *
from test_time_enhancement import SWA


def train_single_fold(fold_names, save_path):
    '''
    train单独一个fold in k_folds CV
    :param fold_names: read in 'foldnames_i_.txt'
    :param save_path:
    :return: save model           save_path------folds----------ckpt-----model_epochs每个epoch的mdoel
                                                       ----------swa_model
                                                       ----------final_model
    '''
    batch_size = 16
    epochs = 12

    fold_number = fold_names.split('_')[-2]
    print('doing fold', fold_number)
    K.clear_session()

    if not os.path.exists(save_path):  # check dir
        os.mkdir(save_path)
    fold_path = save_path + 'fold' + str(fold_number) + '/'
    if not os.path.exists(fold_path):
        os.mkdir(fold_path)
    ckpt_path = fold_path + 'ckpt/'
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    # train
    train_name, validate_name = read_CV_split(fold_names)
    train_ds = prepare_datasets(train_name, label=True, batch_size=batch_size)
    validate_ds = prepare_datasets(validate_name, label=True, batch_size=batch_size)
    model = model_effnet_averagepooling()
    callback_list = [keras.callbacks.ModelCheckpoint(filepath=ckpt_path + 'best_model_fold' + str(fold_number) + '.h5',                                                 monitor='val_auc', save_best_only=True), keras.callbacks.ModelCheckpoint(filepath=ckpt_path + 'fold' + str(fold_number) + '_e{epoch:02d}.h5', save_weights_only=True),keras.callbacks.LearningRateScheduler(lrfn)]
    print('start training')
    model.fit(train_ds, validation_data=validate_ds, epochs=epochs, shuffle=True, validation_freq=1, callbacks=callback_list)
    model.save(fold_path + 'fold' + str(fold_number) + '_final_model.h5')
    # 做SWA
    # swa_model = SWA(ckpt_path, model, start_epoch=13, swa_decay=0.9)#SWA
    # swa_model.save(fold_path+'SWA_fold'+str(fold_number)+'_model.h5')#SWA save

def train_with_cv(train_names, save_path, test_names):
    '''
    包含整个Cross validation过程
    :param train_names: ['str','str'..]
    :param save_path: path=''
    :param test_names: ['str','str'..]
    :return: save model           save_path------folds----------ckpt-----model_epochs每个epoch的mdoel
                                                       ----------swa_model
                                                       ----------final_model
                                            ------folds----------ckpt-----model_epochs每个epoch的mdoel
                                                       ----------swa_model
                                                       ----------final_model
    '''

    if not os.path.exists(save_path):#check dir
      os.mkdir(save_path)

    CV_folds = 5#cross validation fold number
    batch_size = 16
    epochs = 20

  #build prediction df framework
    test_ds = prepare_datasets(test_names, label=False, shuffle=False)
    image_name_ds = test_ds.map(lambda i,j:j)
    image_names = []
    for item in image_name_ds:

        temp = item.numpy()
        for item2 in temp:
            image_names.append(item2.decode('utf-8'))
    image_names = np.array(image_names)
    df_pred = pd.DataFrame({'image_name':image_names})

  #K_folds cross validation:
    kfolds = model_selection.KFold(n_splits=CV_folds, shuffle=True)
    for i, (train_idx, validate_idx) in enumerate(kfolds.split(train_names)):
        K.clear_session()
        fold_path = save_path+'fold'+str(i)+'/'
        if not os.path.exists(fold_path):
            os.mkdir(fold_path)
            ckpt_path = fold_path+'ckpt/'
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

    #train
    train_name = [train_names[index] for index in train_idx]
    validate_name = [train_names[index] for index in validate_idx]
    train_ds = prepare_datasets(train_name, label=True, batch_size=batch_size)
    validate_ds = prepare_datasets(validate_name, label=True, batch_size=batch_size)
    model = model_effnet_averagepooling()
    callback_list = [keras.callbacks.ModelCheckpoint(filepath=ckpt_path+'best_model_fold'+str(i)+'.h5', monitor='val_auc', save_best_only=True),
                     keras.callbacks.ModelCheckpoint(filepath=ckpt_path+'fold'+str(i)+'_e{epoch:02d}.h5', save_weights_only=True),
                     keras.callbacks.LearningRateScheduler(lrfn)]
    print('start fold', i+1)
    print('start training')
    model.fit(train_ds, validation_data=validate_ds, epochs=epochs, shuffle=True, validation_freq=1, callbacks=callback_list)
    model.save(fold_path+'fold'+str(i)+'_final_model.h5')
    #做SWA
    swa_model = SWA(ckpt_path, model, start_epoch=13, swa_decay=0.9)#SWA
    swa_model.save(fold_path+'SWA_fold'+str(i)+'_model.h5')#SWA save

    #每个k_fold自动做test
    #print('start testing')
    #test_ds = prepare_datasets(test_names, label=False, shuffle=False)
    #predictions = model.predict(test_ds.map(lambda x,y:x), verbose=1)
    #print('finsih testing')
    #df_pred['target'+str(i+1)] = predictions.squeeze()
    #df_pred.to_csv(save_path+'targets.csv', index=False)
    #print('finish fold', i+1)

    #每个k_fold自动做TTA test
    #print('start TTA testing')
    #TTA_repeat_number = 4
    #test_ds = prepare_datasets(test_names, repeat=TTA_repeat_number, shuffle=False, label=False, augment=True, batch_size=batch_size)
    #preditions = model.predict(test_ds.map(lambda x,y:x), verbose=1)
    #size = predictions.shape[0]/TTA_repeat_number
    #array_temp = np.ndarray([size, TTA_repeat_number])
    #for j in range(TTA_repeat_number):
    #  array_temp[:,j] = preditions[j*size:(j+1)*size]
    #df_pred['target'+str(i+1)] = np.mean(array_temp, axis=1)
    #df_pred.to_csv(save_path+'targets.csv', index=False)
    #print('finish TTA testing for fold', i+1)

def read_data_name():
    '''
    读入合并train， test data name
    :return: train_names: ['str','str'..], test_names: ['str','str'..]
    '''
    path1 = 'data/melanoma-384x384'
    path2 = 'data/isic2019-384x384'
    path3 = 'data/malignant-v2-384x384'
    filename_orig = tf.io.gfile.glob(os.path.join(path1, "train*.tfrec"))
    filename_2019 = tf.io.gfile.glob([os.path.join(path2, "train%.2i*.tfrec" % i) for i in range(1, 30, 2)])
    filename_2018 = tf.io.gfile.glob([os.path.join(path2, "train%.2i*.tfrec" % i) for i in range(0, 30, 2)])
    filename_mal = tf.io.gfile.glob([os.path.join(path3, 'train%.2i*.tfrec' % i) for i in range(15, 30, 1)])
    filename_test = tf.io.gfile.glob(os.path.join(path1, "test*.tfrec"))

    train_names = filename_orig + filename_2018 + filename_2019 + filename_mal
    test_names = filename_test
    return train_names, test_names

def CV_split(filenames, save_path):
    '''
    #保存CV的split 为了单独train每个fold
    :param filenames: train data name list ['str','str'...]
    :param save_path: path=''
    :return: save 'foldnames_i_.txt'
    '''
    CV_folds = 5
    kfolds = model_selection.KFold(n_splits=CV_folds, shuffle=True)
    for i, (train_idx, validate_idx) in enumerate(kfolds.split(filenames)):
        train_names = [filenames[item] for item in train_idx]
        validate_names = [filenames[item] for item in validate_idx]
        fold = [train_names, validate_names]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        with open(save_path+'foldnames_'+str(i)+'_.txt', 'wb') as f:
            pickle.dump(fold, f)

def read_CV_split(path):
    '''
    读入单个fold，输出train_name， validate_name
    :param path: read in 'foldnames_i_.txt'
    :return: train_names: ['str','str'..], validate_names:['str','str'..]
    '''
    with open(path,'rb') as f:
        temp = pickle.load(f)
    train_names = temp[0]
    validate_names = temp[1]
    return train_names, validate_names


if __name__=='__main__':
    #check_gpu()
    tf.config.experimental.list_physical_devices('GPU')

    #train_names, test_names = read_data_name()
    #save_path = 'test/'  # 这类总模型的save_path
    #train_with_cv(train_names, save_path, test_names)


