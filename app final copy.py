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
      # print("Mean Absolute Error:")

      return mae

      # return mse, rmse, mae



names = ['user_id', 'item_id', 'rating', 'timestime']
path = 'D:\MBKM'
columns = ["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", "unknown", "action", "adventure", "animation", "children's", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
movie_data = pd.read_csv(os.path.join(path, 'Datasets/ml-100k', 'u.item'), sep='|', names=columns,encoding="latin-1",index_col="movie_id")
ratings_train_k1_old = pd.read_csv(os.path.join(path, 'Datasets/ml-100k', 'u1.base'), sep='\t', names=names)
ratings_test_k1 = pd.read_csv(os.path.join(path, 'Datasets/ml-100k', 'u1.test'), sep='\t', names=names)
banyak_users = np.unique(ratings_test_k1["user_id"])
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
# print("===========================================")
# print("sudah berhasil")

# merubah data ke dataframe
# mean_user = pd.DataFrame(mean_user, index=rating_matrix.index)
# mean_center_user = pd.DataFrame(mean_center_user, index=rating_matrix.index, columns=rating_matrix.columns)
# similarity_user = pd.DataFrame(similarity_user, index=rating_matrix.index, columns=rating_matrix.index)

# EVALUASI
def precision(ground_truth, topN, n=1):
    return (len(np.intersect1d(topN[:n], ground_truth)) / n)

def recall(ground_truth, topN, n=1):
    return len(np.intersect1d(topN[:n], ground_truth)) / len(set(ground_truth))

def f1Score(ground_truth, topN, n=1):
    p = precision(ground_truth, topN, n)
    r = recall(ground_truth, topN, n)

    return ((2 * p * r) / (p + r)) if (p > 0 and r > 0) else 0

def idcg(n):
    print("idcg",np.sum((1 / np.log2(1 + np.array(list(range(1, n+1)))))))
    return np.sum((1 / np.log2(1 + np.array(list(range(1, n+1))))))

def dcg(ground_truth, topN, n):
    a = np.array([(1 / np.log2(1 + x)) for x in range(1,n+1)])
    print(a)
#     b = np.array([np.sum(np.where(tp == ground_truth, 1, 0)) for tp in topN[:n]])
#     b = np.array([np.sum(np.where(tp in ground_truth, 1, 0)) for tp in topN[:n]])
    b = np.array([np.sum(np.where(tp == np.array(ground_truth), 1, 0)) for tp in topN[:n]])
    print(b)
    return np.sum(a*b)

def ndcg(ground_truth, topN, n):
    return (dcg(ground_truth, topN, n) / idcg(n))


def evaluasi(ground_truth, topN, n=1):
    """Calculate and show MSE, RMSE, & MAE from predicted rating with hybrid method

    Parameters
    ----------
    y_actual: numpy.ndarray
        The user-item test data
    y_predicted: numpy.ndarray
      The user-item rating that have been predicted with hybrid method

    Returns
    -------
    precision: float
        mean squared error of hybrid method
    recall: float
        root mean squared error of hybrid method
    f1-score: float
        mean absolute error of hybrid method
    """

    return [precision(ground_truth, topN, n=1), recall(ground_truth, topN, n=1), f1Score(ground_truth, topN, n=1), ndcg(ground_truth, topN, n=1)]




data_user = rating_matrix.to_numpy()

@app.route("/")
def index_page():
      navnya=["Home","Rekomendasi Film","Tentang Aplikasi"]
      judulnya = "Rekomendasi System"
      nama_user = "Selamat datang di"
      # films = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
      banyak_user=[]
      for i in banyak_users:
            banyak_user.append(i)
      banyak_n=[]
      for i in range(1,51):
            banyak_n.append(i)
      pesan_error=""
      return render_template("index.html",navnya=navnya, judulnya=judulnya, nama_user=nama_user,banyak_user=banyak_user, banyak_n=banyak_n,pesan_error=pesan_error)

@app.route("/rekomendasi")
def rekomendasi_page():
      navnya=["Home"," Hasil Rekomendasi Film",""]
      judulnya = "Hasil Rekomendasi"
      id_user=int(request.args.get('user'))
      tetangga=int(request.args.get("tetangga"))
      # print(id_user.type())
      # print(tetangga.type())
      # print("==========================="*2)
      # data_ground_truth = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5"]
      index_data_ground_truth=[]
      # print("ini untuk i")
      asem=0
      for i in range(1,len(rating_matrix_test.loc[id_user]+1)):
            if rating_matrix_test.loc[id_user][i]!=0.0:
                  index_data_ground_truth.append(i)
      data_ground_truth=[]
      for i in index_data_ground_truth:
            data_ground_truth.append(movie_data.loc[i][0])
      banyak_data_ground_truth = len(data_ground_truth)
      index_data_train=[]
      # for i in range(len(rating_matrix[id_user])):
      for i in range(1,len(rating_matrix.loc[id_user]+1)):
            if rating_matrix.loc[id_user,i]!=0.0:
                  index_data_train.append(i)
      data_train=[]
      for i in index_data_train:
            data_train.append(movie_data.loc[i][0])
      # index_data_train=rating_matrix.iloc[id_user,:]
      banyak_data_train=len(data_train)
      
      if id_user not in banyak_users:
            # @app.route("/")
            navnya=["Home","Rekomendasi Film","Tentang Aplikasi"]
            judulnya = "Rekomendasi System"
            nama_user = "Selamat datang di"
            films = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
            banyak_user=[]
            for i in banyak_users:
                  banyak_user.append(i)
            banyak_n=[]
            for i in range(1,51):
                  banyak_n.append(i)
            pesan_error="aktif"
            return render_template("index.html",navnya=navnya, judulnya=judulnya, nama_user=nama_user,banyak_user=banyak_user, banyak_n=banyak_n,pesan_error=pesan_error,id_user=id_user)
      
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
                  jenis="user",
                  tetangga=10
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
                  # print(i)
                  user_recomendations.append(movie_norated_test[pred_user.index(i)])
            else:
                  break
            temp+=1
      
      # print("==================================================")
      # print("USER DONE")
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
                  jenis="item",
                  tetangga=100
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

      recomendations = []
      # gt = ratings_test_k1[ratings_test_k1['user_id'] == id_user].loc[:,'item_id'].tolist()
      # topN = hybrid_toy_data[(-hybrid_toy_data[:, 1].astype(float)).argsort()][:,0]
      # evTopN = [[],[],[],[]]
      # for n in range(1, 101):
      #       p = precision(ground_truth=gt, topN=topN, n=n)
      #       r = recall(ground_truth=gt, topN=topN, n=n)
      #       f = f1Score(ground_truth=gt, topN=topN, n=n)
      #       n = ndcg(ground_truth=gt, topN=topN, n=n)
      #       evTopN[0].append(p)
      #       evTopN[1].append(r)
      #       evTopN[2].append(f)
      #       evTopN[3].append(n)

      temp=0
      for i in hybrid_topn:
            if temp<tetangga:
                  recomendations.append(movie_norated_test[hybrid_toy_data.index(i)])
            else:
                  break
            temp+=1
      hasil_rekomendasi=[]
      imdb_film=[]
      for i in recomendations:
            hasil_rekomendasi.append(movie_data.loc[i][0])
            imdb_film.append(movie_data.loc[i][2])
      print("imdb done")        
      count=0
      for i in hasil_rekomendasi:
            if count < tetangga and count<50:
                  top_n.append(i)
            count+=1
      # precision=0.1
      # recall=0.2
      # f1=0.3
      banyak_data_rekomendasi=len(top_n)
      banyak_data_irisan=0
      for i in top_n:
            if i in data_ground_truth:
                  banyak_data_irisan+=1

      # EVALUASI MATRIX
      evTopN = [[],[],[],[],[]]
      # for n in range(1, 101):
      # print("ini data ground truth",data_ground_truth)
      print("================================================================================================================")
      # print("ini data hasil Rekomendasi",hasil_rekomendasi)
      p = precision(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      print("Precision",p)
      r = recall(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      f = f1Score(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      d = dcg(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      nd = ndcg(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      evTopN[0].append(p)
      evTopN[1].append(r)
      evTopN[2].append(f)
      evTopN[3].append(d)
      evTopN[4].append(nd)
      # ev.append(evTopN)

      # tetangga=tetangga
      # get
      banyak_user=[]
      for i in banyak_users:
            banyak_user.append(i)
      for i in range(1,463):
            if i not in banyak_user:
                  print(i)
      nama_user = "User "+str(id_user)
      return render_template("hasilrekomendasi.html",navnya=navnya,judulnya=judulnya,user=id_user,nama_user=nama_user, tetangga=tetangga, #films=films, 
      banyak_data_train=banyak_data_train, data_train=data_train,
      banyak_data_rekomendasi=banyak_data_rekomendasi,
      banyak_data_irisan=banyak_data_irisan,#data_irisan=data_irisan,
      banyak_data_ground_truth=banyak_data_ground_truth,data_ground_truth=data_ground_truth,
      hasil_rekomendasi=hasil_rekomendasi,
      top_n=top_n,imdb_film=imdb_film,
      # precision=precision,recall=recall,f1=f1
      precision=evTopN[0][0],recall=evTopN[1][0],f1=evTopN[2][0],dcg=evTopN[3][0],ndcg=evTopN[4][0]
      )


@app.route("/metrik_evaluasi")
def metrik_evaluasi_page():
      navnya=["Home","","Metrik Evaluasi"]
      judulnya = "Hasil Rekomendasi"
      id_user=int(request.args.get('user'))
      tetangga=int(request.args.get("tetangga"))
      # print(id_user.type())
      # print(tetangga.type())
      # print("==========================="*2)
      # data_ground_truth = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5"]
      index_data_ground_truth=[]
      # print("ini untuk i")
      asem=0
      for i in range(1,len(rating_matrix_test.loc[id_user]+1)):
            if rating_matrix_test.loc[id_user][i]!=0.0:
                  index_data_ground_truth.append(i)
      data_ground_truth=[]
      for i in index_data_ground_truth:
            data_ground_truth.append(movie_data.loc[i][0])
      banyak_data_ground_truth = len(data_ground_truth)
      index_data_train=[]
      # for i in range(len(rating_matrix[id_user])):
      for i in range(1,len(rating_matrix.loc[id_user]+1)):
            if rating_matrix.loc[id_user,i]!=0.0:
                  index_data_train.append(i)
      data_train=[]
      for i in index_data_train:
            data_train.append(movie_data.loc[i][0])
      # index_data_train=rating_matrix.iloc[id_user,:]
      banyak_data_train=len(data_train)
      
      if id_user not in banyak_users:
            # @app.route("/")
            navnya=["Home","Rekomendasi Film","Tentang Aplikasi"]
            judulnya = "Rekomendasi System"
            nama_user = "Selamat datang di"
            films = ["ini film 1","ini film 2","ini film 3","ini film 4","ini film 5","ini film 6","ini film 7","ini film 8","ini film 9","ini film 10",]
            banyak_user=[]
            for i in banyak_users:
                  banyak_user.append(i)
            banyak_n=[]
            for i in range(1,51):
                  banyak_n.append(i)
            pesan_error="aktif"
            return render_template("index.html",navnya=navnya, judulnya=judulnya, nama_user=nama_user,banyak_user=banyak_user, banyak_n=banyak_n,pesan_error=pesan_error,id_user=id_user)
      
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
                  jenis="user",
                  tetangga=10
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
                  # print(i)
                  user_recomendations.append(movie_norated_test[pred_user.index(i)])
            else:
                  break
            temp+=1
      
      # print("==================================================")
      # print("USER DONE")
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
                  jenis="item",
                  tetangga=100
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

      recomendations = []
      # gt = ratings_test_k1[ratings_test_k1['user_id'] == id_user].loc[:,'item_id'].tolist()
      # topN = hybrid_toy_data[(-hybrid_toy_data[:, 1].astype(float)).argsort()][:,0]
      # evTopN = [[],[],[],[]]
      # for n in range(1, 101):
      #       p = precision(ground_truth=gt, topN=topN, n=n)
      #       r = recall(ground_truth=gt, topN=topN, n=n)
      #       f = f1Score(ground_truth=gt, topN=topN, n=n)
      #       n = ndcg(ground_truth=gt, topN=topN, n=n)
      #       evTopN[0].append(p)
      #       evTopN[1].append(r)
      #       evTopN[2].append(f)
      #       evTopN[3].append(n)

      temp=0
      for i in hybrid_topn:
            if temp<tetangga:
                  recomendations.append(movie_norated_test[hybrid_toy_data.index(i)])
            else:
                  break
            temp+=1
      hasil_rekomendasi=[]
      imdb_film=[]
      for i in recomendations:
            hasil_rekomendasi.append(movie_data.loc[i][0])
            imdb_film.append(movie_data.loc[i][2])
      print("imdb done")        
      count=0
      for i in hasil_rekomendasi:
            if count < tetangga and count<50:
                  top_n.append(i)
            count+=1
      # precision=0.1
      # recall=0.2
      # f1=0.3
      banyak_data_rekomendasi=len(top_n)
      banyak_data_irisan=0
      for i in top_n:
            if i in data_ground_truth:
                  banyak_data_irisan+=1

      # EVALUASI MATRIX
      evTopN = [[],[],[],[],[]]
      # for n in range(1, 101):
      # print("ini data ground truth",data_ground_truth)
      print("================================================================================================================")
      # print("ini data hasil Rekomendasi",hasil_rekomendasi)
      p = precision(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      print("Precision",p)
      r = recall(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      f = f1Score(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      d = dcg(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      nd = ndcg(ground_truth=data_ground_truth, topN=hasil_rekomendasi, n=int(tetangga))
      evTopN[0].append(p)
      evTopN[1].append(r)
      evTopN[2].append(f)
      evTopN[3].append(d)
      evTopN[4].append(nd)
      # ev.append(evTopN)

      # tetangga=tetangga
      # get
      banyak_user=[]
      for i in banyak_users:
            banyak_user.append(i)
      for i in range(1,463):
            if i not in banyak_user:
                  print(i)
      nama_user = "User "+str(id_user)
      evTopN[0][0] = round(evTopN[0][0], 4)
      evTopN[1][0] = round(evTopN[1][0], 4)
      evTopN[2][0] = round(evTopN[2][0], 4)
      evTopN[3][0] = round(evTopN[3][0], 4)
      evTopN[4][0] = round(evTopN[4][0], 4)
      # EvTopN =[ round(elem, 3) for elem in my_list ]
      print('evTopN',round(evTopN[4][0], 4))
      return render_template("metrik_evaluasi.html",navnya=navnya,judulnya=judulnya,user=id_user,nama_user=nama_user, tetangga=tetangga, #films=films, 
      banyak_data_train=banyak_data_train, data_train=data_train,
      banyak_data_rekomendasi=banyak_data_rekomendasi,
      banyak_data_irisan=banyak_data_irisan,#data_irisan=data_irisan,
      banyak_data_ground_truth=banyak_data_ground_truth,data_ground_truth=data_ground_truth,
      hasil_rekomendasi=hasil_rekomendasi,
      top_n=top_n,imdb_film=imdb_film,
      # precision=precision,recall=recall,f1=f1
      precision=evTopN[0][0],recall=evTopN[1][0],f1=evTopN[2][0],dcg=evTopN[3][0],ndcg=evTopN[4][0]
      # precision=precision,recall=evTopN[1][0],f1=evTopN[2][0],dcg=evTopN[3][0],ndcg=evTopN[4][0]
      )

      
if __name__ == "__main__":
      app.run(debug=True)