from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import adjusted_mutual_info_score
from IPython.display import HTML
import pickle

app = Flask(__name__)
path = 'D:\MBKM'

# hybrid
def calculate_mean(data):
      user_mean = (data.sum(axis=1))/(np.count_nonzero(data, axis=1))
      user_mean[np.isnan(user_mean)] = 0.0
      return user_mean


def calculate_mean_centered(data, mean):

      mat_mean_centered = []
      # iterate by rows
      for i in range(len(data)):
            row = []
            # iterate columns
            for j in range(len(data[i])):
                  row.append(data[i][j] - mean[i] if data[i][j] != 0 else 0)
            mat_mean_centered.append(row)

      return np.array(mat_mean_centered)


def predict(datas, mean, mean_centered, similarity, user=3, item=2, tetangga=2, jenis='user'):
        
  hasil = 0
  try:
    # determine based model wheter user-based or item-based
    # take user/item rating, mean centered, and simillarity to calculate
    if jenis == "user":
        dt = datas.loc[:, item].to_numpy()
        meanC = mean_centered.loc[:, item].to_numpy()
        simi = similarity.loc[user, :].to_numpy()
    elif jenis == "item":
        try:
            dt = datas.loc[:, user].to_numpy()
            meanC = mean_centered.loc[:, user].to_numpy()
            simi = similarity.loc[item, :].to_numpy()
        except KeyError:
            simi = np.zeros(similarity.shape[1])
            print(f"User {user} has yet rated Item {item}")

    # user/item index that is yet rated
    idx_dt = np.where(dt != 0)

    # filter user/item rating, mean centered, and simillarity value that is not zero
    nilai_mean_c = np.array(meanC)[idx_dt]
    nilai_similarity = simi[idx_dt]

    # take user/item simillarity index as neighbors and sort it
    idx_sim = (-nilai_similarity).argsort()[:tetangga]


    # see equation 5 & 6 (prediction formula) in paper
    # numerator
    a = np.sum(nilai_mean_c[idx_sim] * nilai_similarity[idx_sim])

    # denomerator
    b = np.abs(nilai_similarity[idx_sim]).sum()

    # check denominator is not zero and add μ (mean rating)
    if b != 0:

      if jenis == "user":
          hasil = mean.loc[user] + (a/b)
          if a==0 or b==0:
            hasil=0
      else:
          hasil = mean.loc[item] + (a/b)
          if a==0 or b==0:
            hasil=0

    else:
      if jenis == "user":
          hasil = mean.loc[user] + 0

      else:
          hasil = mean.loc[item] + 0

  except KeyError:
    if jenis == "user":
        print(f"Item {item} has never rated by all users")
        hasil = mean.loc[user] + 0
    else:
        print(f"User {user} has yet rated Item {item}")
        hasil = mean.loc[item] + 0

  return hasil


def hybrid(predict_user, predict_item, r1=0.7):

      # degree of fusion will be splitted in to two parameter
      # the one (Γ1) is used for user-based model
      # the others (Γ2 = 1 - Γ1) is used for item-based model
      r = np.array([r1, 1-r1])

      # weighting all the users and items corresponding to the Topk UCF and TopkICF models
      # see equation 13 (hybrid formula) in paper
      r_caping = np.column_stack((predict_user, predict_item))
      result = np.sum((r*r_caping), axis=1)

      return result


def evaluasi(y_actual, y_predicted):

      # mse is sama as rmse without square root
      # mse = np.square(np.subtract(y_actual,y_predicted)).mean()

      # see equation 15 in paper
      # rmse = np.sqrt(mse)

      # see equation 14 in paper
      mae = np.mean(np.abs(y_actual - y_predicted))

      # print("Mean Square Error:")
      # print(mse)
      # print()
      # print("Root Mean Square Error:")
      # print(rmse)
      # print()
      print("Mean Absolute Error:")

      return mae

      # return mse, rmse, mae



names = ['user_id', 'item_id', 'rating', 'timestime']
path = 'D:\MBKM'
ratings_train_k1_old = pd.read_csv(os.path.join(path, 'Datasets/ml-100k', 'u1.base'), sep='\t', names=names)
ratings_test_k1 = pd.read_csv(os.path.join(path, 'Datasets/ml-100k', 'u1.test'), sep='\t', names=names)
rating_matrix_k1 = pd.DataFrame(np.zeros((943, 1682)), index=list(range(1,944)), columns=list(range(1,1683))).rename_axis(index='user_id', columns="item_id")
# train
rating_matrix_k1_old = ratings_train_k1_old.pivot_table(index='user_id', columns='item_id', values='rating')
rating_matrix_k1_old = rating_matrix_k1_old.fillna(0)
rating_matrix_k1.update(rating_matrix_k1_old)
# calculate contingency matrix
rating_matrix = pd.DataFrame(np.zeros((943, 1682)), index=list(range(1,944)), columns=list(range(1,1683))).rename_axis(index='user_id', columns="item_id")
rating_matrix_test = pd.DataFrame(np.zeros((943, 1682)), index=list(range(1,944)), columns=list(range(1,1683))).rename_axis(index='user_id', columns="item_id")

# load dataset k-fold, train dan test
ratings_train = pd.read_csv(os.path.join(path, f'Datasets/ml-100k/u1.base'), sep='\t', names=names)
ratings_test = pd.read_csv(os.path.join(path, f'Datasets/ml-100k/u1.test'), sep='\t', names=names)

# merubah dataset menjadi data pivot
rating_matrix_ = ratings_train.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
rating_matrix_ = rating_matrix_.fillna(0)
# update data rating dummie
rating_matrix.update(rating_matrix_)
result_rating_matrix=rating_matrix.iloc[:5,:5]
# result_rating_matrix=rating_matrix
result_rating_matrix=HTML(result_rating_matrix.to_html(classes='table table-stripped fortable container')) 
# result = rating_matrix.to_html()

# merubah test menjadi data pivot
rating_matrix_test_ = ratings_test.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
rating_matrix_test_ = rating_matrix_test_.fillna(0)
rating_matrix_test.update(rating_matrix_test_)
result_rating_matrix_test=rating_matrix_test.iloc[:5,:5]
result_rating_matrix_test=HTML(result_rating_matrix_test.to_html(classes='table table-stripped fortable container')) 
# result_rating_matrix=text_file.write(result)

# ===================================================================================================
# Item
rating_matrix_T = rating_matrix.copy().T
with open(os.path.join(path, 'code', 'model', f'item_k11.pkl'), 'rb') as model_file:
            item_mean_user, item_mean_center_user, item_similarity_user  = pickle.load(model_file)
item_mean_user = pd.DataFrame(item_mean_user, index=rating_matrix_T.index)
item_mean_centered_user = pd.DataFrame(item_mean_center_user, index=rating_matrix_T.index, columns=rating_matrix_T.columns)
item_similarity_user = pd.DataFrame(item_similarity_user, index=rating_matrix_T.index, columns=rating_matrix_T.index)


# ===================================================================================================
# USER
with open(os.path.join(path, 'code', 'model', f'user_k11.pkl'), 'rb') as model_file:
            mean_user, mean_center_user, similarity_user  = pickle.load(model_file)
mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)
print("===========================================")
print("sudah berhasil")

# merubah data ke dataframe
# mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
# mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
# similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)

# proses user-based
data_user = rating_matrix.to_numpy()

@app.route("/")
def index_page():
      navnya=["Home","Rekomendasi Film","About"]
      judulnya = "Rekomendasi System"
      user = "Selamat datang di"
      films = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
      banyak_user=[]
      for i in range(1,944):
            banyak_user.append(i)
      banyak_n=[]
      for i in range(1,30):
            banyak_n.append(i)
      return render_template("index.html",navnya=navnya, judulnya=judulnya, films=films, user=user,banyak_user=banyak_user, banyak_n=banyak_n)

@app.route("/rekomendasi")
def rekomendasi_page():
      navnya=["Home"," Hasil Rekomendasi Film","Matrik Evaluasi"]
      judulnya = "Hasil Rekomendasi"
      id_user=int(request.args.get('user'))
      tetangga=int(request.args.get("tetangga"))
      # print(id_user.type())
      # print(tetangga.type())
      print("==========================="*2)
      # data_ground_truth = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5"]
      data_ground_truth=[]
      for i in range(len(rating_matrix_test[id_user])):
            if rating_matrix_test.iloc[id_user,i]!=0.0:
                  data_ground_truth.append(i)

      # data_ground_truth = rating_matrix_test.iloc[id_user,:]
      banyak_data_ground_truth = len(data_ground_truth)
      films = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
      # data_train=["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
      data_train=[]
      for i in range(len(rating_matrix[id_user])):
            if rating_matrix.iloc[id_user,i]!=0.0:
                  data_train.append(i)
      # data_train=rating_matrix.iloc[id_user,:]
      banyak_data_train=len(data_train)
      data_rekomendasi=["ini film 1","ini film 12","ini film 3","ini film 4","ini film 5"]
      banyak_data_rekomendasi=len(data_rekomendasi)
      data_irisan=["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5"]
      banyak_data_irisan=len(data_irisan)
      hasil_rekomendasi=["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
      
      
      
      top_n=[]
      data_used_test=rating_matrix.loc[id_user].to_numpy()
      movie_norated_test=np.where(data_used_test == 0)[0]+1
      movie_norated_test.tolist()
      pred_user_datas = np.array(
      [
            # user,
            predict(
                  rating_matrix,
                  mean_user,
                  mean_center_user,
                  similarity_user,
                  user=id_user,
                  item=item,
                  jenis="user"
            ) for item in movie_norated_test
      ]
      )
      # pred user to list
      pred_user = list(pred_user_datas)
      # sorting user
      
      
      user_topn=pred_user.copy()
      # user_topn=sorted(user_topn,reverse=True)
      user_topn.sort(reverse=True)
      # sorting berdasarkan tetangga
      user_recomendations = []
      # banyak n
      temp=0
      for i in user_topn:
            if temp<tetangga:
                  print(i)
                  user_recomendations.append(movie_norated_test[pred_user.index(i)])
            else:
                  break
            temp+=1
      
      print("==================================================")
      print("USER DONE")
      item_data_used_test=rating_matrix.loc[id_user].to_numpy()
      item_movie_norated_test=np.where(item_data_used_test == 0)[0]+1
      item_movie_norated_test.tolist()

      pred_item_datas = np.array(
      [
            # item,
            predict(
                  rating_matrix.T,
                  item_mean_user,
                  item_mean_centered_user,
                  item_similarity_user,
                  user=id_user,
                  item=item,
                  jenis="item"
            ) for item in movie_norated_test
      ]
      )
      pred_item = list(pred_item_datas)
      item_topn=pred_item.copy()
      item_topn.sort(reverse=True)
      item_recomendations = []
      # banyak n
      temp=0
      for i in item_topn:
            if temp<tetangga:
                  item_recomendations.append(movie_norated_test[pred_item.index(i)])
            else:
                  break
            temp+=1
      
      hybrid_toy_data = list(hybrid(pred_user_datas, pred_item_datas))
      hybrid_topn=hybrid_toy_data.copy()
      hybrid_topn.sort(reverse=True)

      recomendations =[]

      temp=0
      for i in hybrid_topn:
            if temp<tetangga:
                  recomendations.append(movie_norated_test[hybrid_toy_data.index(i)])
            else:
                  break
            temp+=1
      hasil_rekomendasi=recomendations
      print("hasil_rekomendasi",hasil_rekomendasi)         
      count=0
      for i in hasil_rekomendasi:
            if count < tetangga:
                  top_n.append(i)
            count+=1
      precision=0.1
      recall=0.2
      f1=0.3
      # tetangga=tetangga
      # get
      return render_template("hasilrekomendasi.html",navnya=navnya,judulnya=judulnya,user=id_user, tetangga=tetangga, films=films, 
      banyak_data_train=banyak_data_train, data_train=data_train,
      data_rekomendasi=data_rekomendasi, banyak_data_rekomendasi=banyak_data_rekomendasi,
      banyak_data_irisan=banyak_data_irisan,data_irisan=data_irisan,
      banyak_data_ground_truth=banyak_data_ground_truth,data_ground_truth=data_ground_truth,
      hasil_rekomendasi=hasil_rekomendasi,
      top_n=top_n,
      precision=precision,recall=recall,f1=f1
      )
      # return "asu lo {}".format(user)
      
if __name__ == "__main__":
      app.run(debug=True)