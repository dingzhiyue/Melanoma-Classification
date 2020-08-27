import pandas as pd
import numpy as np

def SWA(ckpt_path, model, start_epoch=2, swa_decay=0.9):
  '''
  ex：SWA (epoch2+a*epoch3+a^2*epoch4...)/(1+a+a^2....)
  用ckpt fold里每个epoch的ckpt model做

  input：ckpt_path-ckpt根目录， model-keras model， start_epoch-开始的epoch数
  return swa后的完整model
  '''
  all_names = sorted(os.listdir(ckpt_path))#ckpt下的全文件名
  total_decay = 1.0
  model_names = [item for item in all_names if 'best' not in item]
  model_names = model_names[start_epoch-1:]
  print(model_names)
  model_names = [ckpt_path+item for item in model_names]#ckpt 每个epoch model名
  model.load_weights(model_names[0])
  weights = model.get_weights()#list of np array (all layers)全weights
  model_temp = keras.models.clone_model(model)
  if len(model_names)==1:
    return model
  for i in range(1, len(model_names)):#每个epoch更新
    model_temp.load_weights(model_names[i])
    weights_temp = model_temp.get_weights()
    total_decay += swa_decay**i
    for j in range(len(weights)):#单个epoch中的每层更新
      weights[j] = weights[j] + swa_decay**i * weights_temp[j]
  del model_temp
  for j in range(len(weights)):#所有weights normalize
    weights[j] = weights[j] / total_decay
  model.set_weights(weights)

  return model

def test(filenames, model_path='', save_path=''):
  '''
  单独用saved model test
  filenames: test file name list
  model_path='完整model路径。。。.h5'
  save_path='完整save路径。。。.csv'
  '''
  test_ds = prepare_datasets(filenames, label=False, shuffle=False)
  #image_names = np.array([img_name.numpy().decode("utf-8")
  #                      for img, img_name in iter(test_ds.unbatch())])
  image_ds = test_ds.map(lambda i, j: i)
  image_name_ds = test_ds.map(lambda i,j:j)
  image_names = []
  for item in image_name_ds:
    temp = item.numpy()
    for item2 in temp:
      image_names.append(item2.decode('utf-8'))
  image_names = np.array(image_names)
  model = keras.models.load_model(model_path)
  predictions = model.predict(image_ds, verbose=1)
  print('finsih testing')
  df_pred = pd.DataFrame({'image_name':image_names, 'target': predictions.squeeze()})
  df_pred.to_csv(save_path, index=False)
  print('finish saving data')

def TTA_test(filenames, model_path='', save_path='', TTA_repeat_number=10, batch_size=16):
  '''
  单独用saved model 做TTA test（augmentation做test ensenmble再平均）
  filenames: test file name list
  model_path='完整model路径。。。.h5'
  save_path='完整save路径。。。.csv'
  TTA_repeat_number: dataset.repeat的次数（每次随机做augmentation）
  '''
  print('start TTA testing')
  test_ds = prepare_datasets(filenames, repeat=TTA_repeat_number, shuffle=False, label=False, augment=True, batch_size=batch_size)
  image_ds = test_ds.map(lambda i, j: i)
  image_name_ds = test_ds.map(lambda i,j:j)
  image_names = []
  for item in image_name_ds:
    temp = item.numpy()
    for item2 in temp:
      image_names.append(item2.decode('utf-8'))
  image_names = np.array(image_names)
  print(image_names.shape)
  model = keras.models.load_model(model_path)
  predictions = model.predict(test_ds.map(lambda x,y:x), verbose=1)
  size = int(predictions.shape[0]/TTA_repeat_number)
  print(size)
  array_temp = np.ndarray([size, TTA_repeat_number])
  for j in range(TTA_repeat_number):
    array_temp[:,j] = np.squeeze(predictions[j*size:(j+1)*size])

  print('finish TTA testing')
  df_pred = pd.DataFrame({'image_name':image_names[:size], 'target': np.mean(array_temp, axis=1)})
  df_pred.to_csv(save_path, index=False)
  print('finish saving data')

if __name__=='__main__':
  model_path = 'test/model.h5'
  save_path = 'test/TTAprediction.csv'
  # test(filename_test, model_path, save_path)
  df = TTA_test(filename_test, model_path, save_path, TTA_repeat_number=10)