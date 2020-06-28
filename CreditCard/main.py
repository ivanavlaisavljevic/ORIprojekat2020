import hdbscan
from sklearn.manifold import TSNE
import csv
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    data = []
    row_num = 0
    ct = 0
    with open('credit_card_data.csv', mode='r') as csv_file:
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


    projection = TSNE().fit_transform(data)
    # plt.scatter(*projection.T)
    # plt.show()

    print("[CLUSTERING ...]")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(data)
    color_palette = sns.color_palette('Paired',1000)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, clusterer.probabilities_)]
    plt.scatter(*projection.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.show()
