import json
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse.data import _data_matrix
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm

def train(train_set:list, 
          val_set:list, 
          n_epochs:int, 
          relevant_features:list,
          epoch_set_size:int,
          train_val_ratio:tuple,
          save_prefix:str,
          num_boost_round:int=10,
          booster:str="gbtree",
          load_checkpoint:str=None, 
          start_ep:int=0, 
          past_stats:list=None):
  
  params = {
      "tree_method": "approx",
      "booster": booster
  }
  model = None
  stats = []

  if load_checkpoint:
    if start_ep <= 0:
      raise ValueError("please input a valid int for start_ep (>1)")
    if len(past_stats) != start_ep - 1:
      raise ValueError(f"len(past_stats) is {len(past_stats)} but should be start_ep-1, which is {start_ep - 1}")
    model=xgb.Booster()
    model.load_model(load_checkpoint)
    params.update({'process_type': 'update',
                    'updater'     : 'refresh',
                    'refresh_leaf': True,
                    'verbosity': 0})
    stats = past_stats
  
  for epoch_no in range(1, n_epochs+1):
    epoch_train = random.choices(train_set, k=epoch_set_size*train_val_ratio[0])
    epoch_val = random.choices(val_set, k=epoch_set_size*train_val_ratio[1])
    print(f"Epoch {epoch_no}:")
    if epoch_no == 2:
      params.update({'process_type': 'update',
                    'updater'     : 'refresh',
                    'refresh_leaf': True,
                    'verbosity': 0})
    for train_file in tqdm(epoch_train):
      train_df = pd.read_feather(train_file)
      x, y, feature_names = _extract_x_y_featurenames(train_df, 
                                                      relevant_features)
      # y = train_df["Retweets"].values
      # x = train_df[relevant_features]
      # feature_names = x.columns
      # x = x.values
      dtrain = xgb.DMatrix(x, 
                           label=y, 
                           feature_names=feature_names)
      model = xgb.train(params, 
                        dtrain, 
                        num_boost_round=num_boost_round, 
                        xgb_model=model)
    model.save_model(f"{save_prefix}_ep{epoch_no}.model")
    overall_msle = 0
    count = 0
    for val_file in tqdm(epoch_val):
      val_df = pd.read_feather(val_file)
      x, y, feature_names = _extract_x_y_featurenames(val_df, 
                                                      relevant_features)
      # y = val_df["Retweets"].values
      # x = val_df[relevant_features]
      # feature_names = x.columns
      # x = x.values
      y_pred = predict(model, x, feature_names)
      # dval = xgb.DMatrix(x, feature_names=feature_names)
      # y_pred = model.predict(dval)
      # y_pred = y_pred.clip(min=0)
      # y = np.nan_to_num(y)
      msle = mean_squared_log_error(y, y_pred)
      count += 1
      overall_msle += msle
    overall_msle /= count
    print(f"MSLE: {overall_msle}")
    stats.append(overall_msle)
  return model, stats

def load_model(checkpt_path):
  model=xgb.Booster()
  model.load_model(checkpt_path)
  return model

def predict(model:xgb.Booster, 
            x:np.ndarray, 
            feature_names:list=None):
  _data_matrix = xgb.DMatrix(x, feature_names=feature_names)
  y_pred = model.predict(_data_matrix)
  y_pred = y_pred.clip(min=0)
  y_pred = np.nan_to_num(y_pred)
  return y_pred

def _extract_x_y_featurenames(df, relevant_features):
  y = df["Retweets"].values
  x = df[relevant_features]
  feature_names = x.columns
  x = x.values
  y = np.nan_to_num(y)
  return x, y, feature_names

def load_xgboost_prediction(df):
  model = load_model("./models/model/xgboost.model")
  with open("./models/model/relevant_features.json") as f:
    relevant_features = json.load(f)
  x, y, feature_names = _extract_x_y_featurenames(df, relevant_features)
  #x = np.reshape(x,(1, x.size))
  prediction = predict(model, x)
  return prediction

if __name__ == "__main__":
  model = load_model("xgboost.model") # replace model file path
  df = pd.read_feather("../../data/data_188489.ftr") # replace data file path
  with open("relevant_features.json") as f:
    relevant_features = json.load(f)
  x, y, feature_names = _extract_x_y_featurenames(df, relevant_features)
  #x = x[1] # note index of row which we are interested in
  print(y)
  #x = np.reshape(x,(1, x.size))
  prediction = predict(model, x)
  print(prediction)
  print(prediction.shape)
