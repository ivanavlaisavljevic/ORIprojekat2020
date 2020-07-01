from sklearn.cluster import KMeans
import numpy as np
import csv
from matplotlib import pyplot as plt
import pandas

from IPython.display import HTML, display


from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.datasets.loaders import load_nfl
from sklearn.tree import _tree, DecisionTreeClassifier
import seaborn as sns
sns.set()
def kmeans_elbow_viz(data):
    sum_of_squared_distance = []
    n_cluster = range(4, 12)

    for k in n_cluster:
        kmean_model = KMeans(n_clusters=k)
        kmean_model.fit(data)
        sum_of_squared_distance.append(kmean_model.inertia_)

    plt.plot(n_cluster, sum_of_squared_distance, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.title('Elbow method for optimal K')
    plt.show()


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")).data)


def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=[]):
        # feature[i] holds the feature to split on, for the internal node i.
        split_feature = inner_tree.feature[node_id]
        if split_feature != _tree.TREE_UNDEFINED:  # internal node
            name = feature_names[split_feature]
            threshold = inner_tree.threshold[node_id]
            # left child
            left_rule = current_rule + ["({} <= {})".format(name, threshold)]
            tree_dfs(inner_tree.children_left[node_id], left_rule)
            # right child
            right_rule = current_rule + ["({} > {})".format(name, threshold)]
            tree_dfs(inner_tree.children_right[node_id], right_rule)
        else:  # leaf
            dist = inner_tree.value[node_id][0]
            dist = dist / dist.sum()
            max_idx = dist.argmax()
            if len(current_rule) == 0:
                rule_string = "ALL"
            else:
                rule_string = " and ".join(current_rule)
            # register new rule to dictionary
            selected_class = classes[max_idx]
            class_probability = dist[max_idx]
            class_rules = class_rules_dict.get(selected_class, [])
            class_rules.append((rule_string, class_probability))
            class_rules_dict[selected_class] = class_rules

    tree_dfs()  # start from root, node_id = 0
    return class_rules_dict


def cluster_report(data: pandas.DataFrame, clusters, min_samples_leaf=50, pruning_level=0.01):
    # Create Model
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(data, clusters)

    # Generate Report
    feature_names = data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))

    cluster_instance_df = pandas.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pandas.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pandas.merge(cluster_instance_df, report_df, on='class_name', how='left')
    pretty_print(report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']])


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
    data = np.array(data)#.astype(np.float)

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
    # kmeans_elbow_viz(allData)

    #ELBOW ALGORITHM
    #
    # print("[ ELBOW ALGORITHM ... ]")
    #
    # # kelbow_visualizer(KMeans(random_state=4), data, k=(4,12))
    # #
    # model = KMeans()
    # visualizer = KElbowVisualizer(model, k=(4, 12))
    #
    # visualizer.fit(data)  # Fit the data to the visualizer
    # visualizer.show()
    print("[ CLUSTERING ... ]")
    kmeans = KMeans(n_clusters=4, random_state=0, init="k-means++", max_iter=300, n_init=10)
    y_kmeans = kmeans.fit_predict(data)
    X = data
    dataX = np.matrix(X.astype(float))
    dfX = pandas.DataFrame(dataX)
    # first_cluster = np.matrix(X[y_kmeans == 0]).astype(float)
    #
    # print(first_cluster)
    # df_cluster1 = pandas.DataFrame(first_cluster)
    #
    # # # 6 Visualising the clusters
    # # plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=10, c='red', label='Cluster 1')
    # # plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=10, c='blue', label='Cluster 2')
    # # plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=10, c='green', label='Cluster 3')
    # # plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=10, c='cyan', label='Cluster 4')
    # # plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=10, c='magenta', label='Cluster 5')
    # # plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s=10, c='purple', label='Cluster 6')
    # # plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s=10, c='black', label='Cluster 7')
    # #
    # # # Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
    # # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='yellow', label='Centroids')
    # # plt.title('Clusters of CreditCards users')
    # #
    # # plt.show()
    #
    # hist = df_cluster1.hist()
    #
    # plt.title('BALANCE ' + '[ ' + str(len(first_cluster)) + ' ] users')
    # plt.show()

    cluster_report(dfX, y_kmeans, min_samples_leaf=25, pruning_level=0.1)