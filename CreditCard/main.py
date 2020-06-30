from sklearn.cluster import KMeans
import numpy as np
import csv
from matplotlib import pyplot as plt


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
    kmeans = KMeans(n_clusters=8, random_state=0, init="k-means++", max_iter=300, n_init=10)
    y_kmeans = kmeans.fit_predict(data)
    X = data
    # 6 Visualising the clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=10, c='red', label='Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=10, c='blue', label='Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=10, c='green', label='Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=10, c='cyan', label='Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=10, c='magenta', label='Cluster 5')
    plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=10, c='purple', label='Cluster 6')
    plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s=10, c='black', label='Cluster 7')
    plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s=10, c='pink', label='Cluster 8')
    # Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='yellow', label='Centroids')
    plt.title('Clusters of CreditCards users')

    plt.show()

