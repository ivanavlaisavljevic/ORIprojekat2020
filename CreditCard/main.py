from sklearn.cluster import KMeans
import numpy as np
import csv
from matplotlib import pyplot as plt
import pandas
from IPython.display import HTML, display
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.datasets.loaders import load_nfl
import seaborn as sb

def eda(data):
    print("Started exploratory data analysis!")
    k_means = KMeans(n_clusters=6)

    cols = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS"]
    selected = pandas.DataFrame(data[cols])

    label = k_means.fit_predict(selected)

    # 'cluster' column
    selected['cluster'] = label
    cols.append('cluster')

    # Seaborn pairplot
    sb.countplot(data=selected, hue='cluster', x='cluster')
    sb.pairplot(selected[cols], hue='cluster', vars=cols, palette='Dark2')
    plt.show()

    cols1 = ["MONTHLY_AVG_PURCHASE", "MONTHLY_CASH_ADVANCE", "LIMIT_RATIO", "PAYMENT_MIN_RATIO"]
    selected1 = pandas.DataFrame(data[cols1])

    label1 = k_means.fit_predict(selected1)

    # 'cluster' column
    selected1['cluster'] = label1
    cols1.append('cluster')

    # Seaborn pairplot
    sb.countplot(data=selected1, hue='cluster', x='cluster')
    sb.pairplot(selected1[cols1], hue='cluster', vars=cols1, palette='Dark2')
    plt.show()

if __name__ == '__main__':

    data = []
    row_num = 0
    with open("credit_card_data.csv", mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            row = row[1:]
            if row_num == 0:
                row_num = row_num + 1
                continue
            else:
                for item in row:
                    if item == "":
                        row[row.index(item)] = 0
                [float(row[row.index(i)]) for i in row]
                data.append(row)
    data = np.array(data)

    allData = pandas.read_csv("credit_card_data.csv")
    allData['BALANCE'] = allData['BALANCE'].fillna(0)
    allData['BALANCE_FREQUENCY'] = allData['BALANCE_FREQUENCY'].fillna(0)
    allData['PURCHASES'] = allData['PURCHASES'].fillna(0)
    allData['ONEOFF_PURCHASES'] = allData['ONEOFF_PURCHASES'].fillna(0)
    allData['INSTALLMENTS_PURCHASES'] = allData['INSTALLMENTS_PURCHASES'].fillna(0)
    allData['CASH_ADVANCE'] = allData['CASH_ADVANCE'].fillna(0)
    allData['PURCHASES_FREQUENCY'] = allData['PURCHASES_FREQUENCY'].fillna(0)
    allData['ONEOFF_PURCHASES_FREQUENCY'] = allData['ONEOFF_PURCHASES_FREQUENCY'].fillna(0)
    allData['PURCHASES_INSTALLMENTS_FREQUENCY'] = allData['PURCHASES_INSTALLMENTS_FREQUENCY'].fillna(0)
    allData['CASH_ADVANCE_FREQUENCY'] = allData['CASH_ADVANCE_FREQUENCY'].fillna(0)
    allData['CASH_ADVANCE_TRX'] = allData['CASH_ADVANCE_TRX'].fillna(0)
    allData['PURCHASES_TRX'] = allData['PURCHASES_TRX'].fillna(0)
    allData['CREDIT_LIMIT'] = allData['CREDIT_LIMIT'].fillna(0)
    allData['PAYMENTS'] = allData['PAYMENTS'].fillna(0)
    allData['MINIMUM_PAYMENTS'] = allData['MINIMUM_PAYMENTS'].fillna(0)
    allData['TENURE'] = allData['TENURE'].fillna(0)
    allData['PRC_FULL_PAYMENT'] = allData['PRC_FULL_PAYMENT'].fillna(0)
    # allData.info()

    numerical = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
                 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']

    categorical = ['CUST_ID']

    allData = allData[numerical]

    #eda(allData)

    #ELBOW ALGORITHM
    #
    # print("[ ELBOW ALGORITHM ... ]")
    #
    # model = KMeans()
    # visualizer = KElbowVisualizer(model, k=(4, 12))
    #
    # visualizer.fit(allData)  # Fit the data to the visualizer
    # visualizer.show()

    print("[ CLUSTERING ... ]")
    kmeans = KMeans(n_clusters=6, random_state=0, init="k-means++", max_iter=300, n_init=10)
    y_kmeans = kmeans.fit_predict(data)
    X = data

    first_cluster = np.matrix(X[y_kmeans == 3]).astype(float)
    df_cluster1 = pandas.DataFrame(first_cluster)

    # 6 Visualising the clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=10, c='red', label='Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=10, c='blue', label='Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=10, c='green', label='Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=10, c='cyan', label='Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=10, c='magenta', label='Cluster 5')
    plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=10, c='purple', label='Cluster 6')

    # Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='yellow', label='Centroids')
    plt.title('Clusters of CreditCards users')

    plt.show()


    df_cluster1[0].hist(weights=(np.ones_like(df_cluster1.index) / len(df_cluster1.index))*100)

    #sb.set()
    #sns.jointplot(x=df_cluster1[12], y=df_cluster1[2])
    #sb.pairplot(allData, vars=[0, 2])

    plt.title('['+str(len(first_cluster))+']')
    plt.show()
