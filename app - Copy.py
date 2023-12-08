from flask import Flask,render_template
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import adjusted_mutual_info_score

app = Flask(__name__)

@app.route("/")
def index_page():
      # names = ['user_id', 'item_id', 'rating', 'timestime']
      # path = 'D:\MBKM'
      # ratings_train_k1_old = pd.read_csv(os.path.join(path, 'Datasets/ml-100k', 'u1.base'), sep='\t', names=names)
      # ratings_test_k1 = pd.read_csv(os.path.join(path, 'Datasets/ml-100k', 'u1.test'), sep='\t', names=names)
      # rating_matrix_k1 = pd.DataFrame(np.zeros((943, 1682)), index=list(range(1,944)), columns=list(range(1,1683))).rename_axis(index='user_id', columns="item_id")
      # # train
      # rating_matrix_k1_old = ratings_train_k1_old.pivot_table(index='user_id', columns='item_id', values='rating')
      # rating_matrix_k1_old = rating_matrix_k1_old.fillna(0)
      # rating_matrix_k1.update(rating_matrix_k1_old)
      # # calculate contingency matrix
      # def calculate_contingency_matrix(item1, item2):
      #       unique_item1 = np.unique(item1)
      #       unique_item2 = np.unique(item2)
      #       num_true_labels = len(unique_item1)
      #       num_pred_labels = len(unique_item2)
      #       contingency_matrix = np.zeros((num_true_labels, num_pred_labels), dtype=int)
      #       for i, true_label in enumerate(unique_item1):
      #             for j, pred_label in enumerate(unique_item2):
      #                   contingency_matrix[i, j] = np.sum(np.logical_and(item1 == true_label, item2 == pred_label))

      #       return contingency_matrix

      # # calculate entropy
      # def calculate_entropy(item):
      #       unix, counts = np.unique(item, return_counts=True)
      #       probs = counts / len(item)
      #       entropy = -np.sum(probs * np.log(probs))
      #       return entropy

      # # calculate mutual information
      # def calculate_mutual_information(item1, item2):
      #       # contingency_matrix = np.histogram2d(item1, item2, bins=(len(np.unique(item1)), len(np.unique(item2))))[0]
      #       contingency_matrix = calculate_contingency_matrix(item1, item2)
      #       contingency_matrix = contingency_matrix / np.sum(contingency_matrix)
      #       mi = 0
      #       for i in range(contingency_matrix.shape[0]):
      #             for j in range(contingency_matrix.shape[1]):
      #                   if contingency_matrix[i, j] > 0:
      #                         mi+=(contingency_matrix[i, j] * np.log(contingency_matrix[i, j] / (np.sum(contingency_matrix[i, :]) * np.sum(contingency_matrix[:, j]))))

      #       return mi

      # # calculate expected mutual information
      # def calculate_expected_mutual_information(contigency, n_sampel):
      #       n_rows, n_cols = contigency.shape
      #       a = np.ravel(contigency.sum(axis=1).astype(np.int64, copy=False))
      #       b = np.ravel(contigency.sum(axis=0).astype(np.int64, copy=False))
      #       emi = 0
      #       for i in range(n_rows):
      #                   for j in range(n_cols):
      #                         start = max(1, a[i] + b[j] - n_sampel)
      #                         end = min(a[i], b[j]) + 1
      #                         for nij in range(start, end):
      #                               term = (nij/n_sampel)*(np.log((n_sampel*nij)/(a[i]*b[j])))
      #                               atas = (np.math.factorial(a[i])*np.math.factorial(b[j])*np.math.factorial(n_sampel-a[i])*np.math.factorial(n_sampel-b[j]))
      #                               bawah = (np.math.factorial(n_sampel)*np.math.factorial(nij)*np.math.factorial(a[i]-nij)*np.math.factorial(b[j]-nij)*np.math.factorial(n_sampel-a[i]-b[j]+nij))

      #                               emi += term*(atas/bawah)
      #       return emi

      # # calculate adjusted mutual information
      # def calculate_adjusted_mutual_information(item1, item2):
      #       mi = calculate_mutual_information(item1, item2)
      #       entropy_item1 = calculate_entropy(item1)
      #       entropy_item2 = calculate_entropy(item2)
      #       expected_mi = calculate_expected_mutual_information(calculate_contingency_matrix(item1, item2), item1.shape[0])


      #       ami = (mi - expected_mi) / (np.mean([entropy_item1, entropy_item2]) - expected_mi)
      #       return ami

      # def calculate_similarity_ami(data):
      #       """Calculate simmilarity of user-item matrix

      #       Parameters
      #       ----------
      #       data: numpy.ndarray
      #             The user-item matrix to calculate which matrix's size is user (n) times items (m)
      #       mean_centered: numpy.ndarray
      #             the mean centered rating of user-item matrix

      #       Returns
      #       -------
      #       mat_sim: numpy.ndarray
      #             simmilarity of user-item matrix
      #       """
      #       # data = data.to_numpy()
      #       # create empty matrix as placeholder for similarity matrix
      #       mat_sim = [[i for i in range(len(data))] for _ in range(len(data))]

      #       for i in range(len(data)):  # user/item
      #             for j in range(i, len(data)):  # user/item

      #                   # give value of 1 if position of user is same (diagonal)
      #                   if i == j:
      #                         mat_sim[i][j] = 1.0
      #                   else:
      #                         ami = adjusted_mutual_info_score(data[i], data[j])
      #                         if ami > 0:
      #                               mat_sim[i][j] = ami
      #                               mat_sim[j][i] = ami
      #                         else:
      #                               mat_sim[i][j] = ami
      #                               mat_sim[j][i] = ami
      #       return np.array(mat_sim)

      # def similarity(data, mean_centered, means):
      #       """Calculate simmilarity user-item matrix

      #             Parameters
      #             ----------
      #             data: numpy.ndarray
      #                   The user-item matrix to calculate which matrix's size is user (n) times items (m)
      #             mean_centered: numpy.ndarray
      #                   the mean centered rating of user-item matrix

      #             Returns
      #             -------
      #             mat_sim: numpy.ndarray
      #                   simmilarity of user-item matrix
      #             """

      #       # create empty matrix as placeholder for similarity matrix
      #       mat_sim = [[i for i in range(len(data))] for _ in range(len(data))]

      #       for i in range(len(data)):  # user/item
      #             for j in range(i, len(data)):  # user/item

      #                   # give value of 1 if position of user is same (diagonal)
      #                   if i == j:
      #                         mat_sim[i][j] = 1.0
      #                   else:
      #                   # list of mean centered rating of user i is r1 and user j is r2 if rating is not 0
      #                   # e.g. see manual calculation spreadsheet 'user' or 'item' row 16, 17
      #                   # these list have same length
      #                         r1 = data[i]
      #                         r2 = data[j]

      #                         # print("ini i",i,"ini j",j)
      #                         # print(r1, r2)
      #                         # see equation 3 & 4 (use Pearson Coefficient Correlation) in paper
      #                         # if len(r1)!=0:
      #                         pembilang_triagle = np.sqrt(sum(np.square(r1-r2)))
      #                         penyebut_triagle = np.sqrt(sum(np.square(r1))) + np.sqrt(np.sum(np.square(r2)))
      #                         simtriagle = 1 - (pembilang_triagle/penyebut_triagle)
      #                         print(simtriagle)

      #                         ror1 = np.sqrt(sum([k**2 for k in mean_centered[i] if k != 0])/np.count_nonzero(data[i]))
      #                         ror2 = np.sqrt(sum([k**2 for k in mean_centered[j] if k != 0])/np.count_nonzero(data[j]))
      #                         simurp = 1 - (1/(1+(np.exp(-abs(means[i]-means[j])*abs(ror1-ror2)))))

      #                         mat_sim[i][j] = simtriagle*simurp
      #                         mat_sim[j][i] = simtriagle*simurp
      #                         # else:
      #                         #     mat_sim[i][j] = -1
      #                         #     mat_sim[j][i] = -1
      #                         #     # mat_sim[i][j] = 0
      #                         #     # mat_sim[j][i] = 0
      #       return np.array(mat_sim)

      # def calculate_mean(data):
      #       """Calculate mean rating of user-item matrix

      # Parameters
      # ----------
      # data: numpy.ndarray
      #       The user-item matrix to calculate which matrix's size is user (n) times items (m)

      # Returns
      # -------
      # user_mean: numpy.ndarray
      #       mean rating user-item matrix which matrix size is user/items times 1
      # """

      # # sum all rating based on user/item row then divide by number of rating that is not zero
      # user_mean = (data.sum(axis=1))/(np.count_nonzero(data, axis=1))
      # user_mean[np.isnan(user_mean)] = 0.0
      # return user_mean


      # def calculate_mean_centered(data, mean):
      #       """Calculate mean centered rating of user-item matrix

      #       Parameters
      #       ----------
      #       data: numpy.ndarray
      #             The user-item matrix to calculate which matrix's size is user (n) times items (m)
      #       mean: numpy.ndarray
      #             The mean rating of user-item matrix

      #       Returns
      #       -------
      #       mat_mean_centered: numpy.ndarray
      #             mean centered rating user-item matrix which matrix size is same as data parameter
      #       """

      #       mat_mean_centered = []
      #       # iterate by rows
      #       for i in range(len(data)):
      #             row = []
      #             # iterate columns
      #             for j in range(len(data[i])):
      #                   row.append(data[i][j] - mean[i] if data[i][j] != 0 else 0)
      #             mat_mean_centered.append(row)

      #       return np.array(mat_mean_centered)


      # def predict(datas, mean, mean_centered, similarity, user=3, item=2, tetangga=2, jenis='user'):
      #       """Calculate prediction of user target, item target and how many prediction based on number of neighbor

      #       Parameters
      #       ----------
      #       data: numpy.ndarray
      #             The user-item matrix to calculate which matrix's size is user (n) times items (m)
      #       mean: numpy.ndarray
      #             The mean rating of user-item matrix
      #       mean_centered: numpy.ndarray
      #             The mean centered rating of user-item matrix
      #       similarity: numpy.ndarray
      #             The simmalirity (user or item) matrix
      #       user: int
      #             User target, default=3
      #       item: int
      #             Item target, default=2
      #       tetangga: int
      #             Amoun of neighbors, default=2
      #       jenis: str
      #             User or Item based model in CF technique that will be used, default=user

      #       Returns
      #       -------
      #       hasil: numpy.ndarray which has same size as 'tetangga' parameter
      #             simmilarity of user-item matrix
      #       """

      # # determine based model wheter user-based or item-based
      # # take user/item rating, mean centered, and simillarity to calculate
      #       if jenis == "user":
      #             dt = datas.loc[:, item].to_numpy()
      #             meanC = mean_centered.loc[:, item].to_numpy()
      #             simi = similarity.loc[user, :].to_numpy()
      #       elif jenis == "item":
      #             dt = datas.loc[:, user].to_numpy()
      #             meanC = mean_centered.loc[:, user].to_numpy()
      #             simi = similarity.loc[item, :].to_numpy()

      #       # user/item index that has rated
      #       idx_dt = np.where(dt != 0)

      #       # filter user/item rating, mean centered, and simillarity value that is not zero
      #       nilai_mean_c = np.array(meanC)[idx_dt]
      #       nilai_similarity = simi[idx_dt]

      #       # take user/item similarity index as neighbors and sort it
      #       idx_sim = (-nilai_similarity).argsort()[:tetangga]

      #       # see equation 5 & 6 (prediction formula) in paper
      #       # numerator
      #       a = np.sum(nilai_mean_c[idx_sim] * nilai_similarity[idx_sim])

      #       # denomerator
      #       b = np.abs(nilai_similarity[idx_sim]).sum()

      #       # check denominator is not zero and add μ (mean rating)
      #       if b != 0:
      #             if jenis == "user":
      #                   hasil = mean.loc[user] + (a/b)
      #             else:
      #                   hasil = mean.loc[item] + (a/b)
      #       else:
      #             if jenis == "user":
      #                   hasil = mean.loc[user] + 0
      #             else:
      #                   hasil = mean.loc[item] + 0

      #       return [item, float(hasil)]


      # def hybrid(predict_user, predict_item, r1=0.7):
      #       """Calculate prediction of user-item matrix from hybridization of collaborative learning with Liniear Regression

      #       Parameters
      #       ----------
      #       data: numpy.ndarray
      #             The user-item matrix to calculate which matrix's size is user (n) times items (m)
      #       predict_user: numpy.ndarray
      #             The prediction ratings of user
      #       predict_item: numpy.ndarray
      #             The prediction ratings of item
      #       r1: int
      #             degree fusion is used as the weights of the prediction function (see Equations 12 and 13) in paper

      #       Returns
      #       -------
      #       result: numpy.ndarray
      #             unknown list of prediction with hybrid method
      #       """

      #       # degree of fusion will be splitted in to two parameter
      #       # the one (Γ1) is used for user-based model
      #       # the others (Γ2 = 1 - Γ1) is used for item-based model
      #       r = np.array([r1, 1-r1])

      #       # weighting all the users and items corresponding to the Topk UCF and TopkICF models
      #       # see equation 13 (hybrid formula) in paper
      #       r_caping = np.column_stack((predict_user, predict_item))
      #       result = np.sum((r*r_caping), axis=1)

      #       return result


      # def evaluasi(y_actual, y_predicted):
      #       """Calculate and show MSE, RMSE, & MAE from predicted rating with hybrid method

      #       Parameters
      #       ----------
      #       y_actual: numpy.ndarray
      #             The user-item test data
      #       y_predicted: numpy.ndarray
      #             The user-item rating that have been predicted with hybrid method

      #       Returns
      #       -------
      #       mse: float
      #             mean squared error of hybrid method
      #       rmse: float
      #             root mean squared error of hybrid method
      #       mae: float
      #             mean absolute error of hybrid method
      #       """

      #       # mse is sama as rmse without square root
      #       # mse = np.square(np.subtract(y_actual,y_predicted)).mean()

      #       # see equation 15 in paper
      #       # rmse = np.sqrt(mse)

      #       # see equation 14 in paper
      #       mae = np.mean(np.abs(y_actual - y_predicted))

      #       # print("Mean Square Error:")
      #       # print(mse)
      #       # print()
      #       # print("Root Mean Square Error:")
      #       # print(rmse)
      #       # print()
      #       print("Mean Absolute Error:")

      #       return mae

      # # return mse, rmse, mae


      # predicted_ratings_user = np.array(
      #       [
      #             predict(
      #                   rating_matrix_k1,
      #                   mean_user_df_k1,
      #                   mean_centered_user_df_k1,
      #                   similarity_user_df_k1,
      #                   item=item,
      #                   user=user,
      #                   tetangga=30
      #             ) for (item,user) in zip(ratings_test_k1['item_id'], ratings_test_k1['user_id'])
      #       ]
      # )



      judulnya = "Rekomendasi isi"
      return render_template("index.php",judulnya=judulnya)
if __name__ == "__main__":
      app.run(debug=True)