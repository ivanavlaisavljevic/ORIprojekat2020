from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import pandas
import pandas as pd
from sklearn.tree import _tree, DecisionTreeClassifier
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")


def get_class_rules(tree: DecisionTreeClassifier, feature_names: list):
    inner_tree: _tree.Tree = tree.tree_
    classes = tree.classes_
    class_rules_dict = dict()

    def tree_dfs(node_id=0, current_rule=None):
        # feature[i] holds the feature to split on, for the internal node i.
        if current_rule is None:
            current_rule = []
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


def cluster_report(the_data: pd.DataFrame, cluster_num, min_samples_leaf=45, pruning_level=0.02):
    print("[ MAKING CLUSTER REPORT ... ]")
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(the_data)
    km = KMeans(n_clusters=cluster_num)
    clusters = km.fit_predict(data_pca)

    # Create Model
    tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, ccp_alpha=pruning_level)
    tree.fit(the_data, clusters)

    # Generate Report
    feature_names = the_data.columns
    class_rule_dict = get_class_rules(tree, feature_names)

    report_class_list = []
    for class_name in class_rule_dict.keys():
        rule_list = class_rule_dict[class_name]
        combined_string = ""
        for rule in rule_list:
            combined_string += "[{}] {}\n\n".format(rule[1], rule[0])
        report_class_list.append((class_name, combined_string))

    cluster_instance_df = pd.Series(clusters).value_counts().reset_index()
    cluster_instance_df.columns = ['class_name', 'instance_count']
    report_df = pd.DataFrame(report_class_list, columns=['class_name', 'rule_list'])
    report_df = pd.merge(cluster_instance_df, report_df, on='class_name', how='left')
    (report_df.sort_values(by='class_name')[['class_name', 'instance_count', 'rule_list']]).to_csv("clusters_reports.csv")


def load_data():
    print("[ LOADING DATA ... ]")
    the_data = pandas.read_csv("credit_card_data.csv")

    the_data['BALANCE'] = the_data['BALANCE'].fillna(the_data['BALANCE'].median())
    the_data['PURCHASES'] = the_data['PURCHASES'].fillna(the_data['PURCHASES'].median())
    the_data['CASH_ADVANCE'] = the_data['CASH_ADVANCE'].fillna(the_data['CASH_ADVANCE'].median())
    the_data['CREDIT_LIMIT'] = the_data['CREDIT_LIMIT'].fillna(the_data['CREDIT_LIMIT'].median())
    the_data['PAYMENTS'] = the_data['PAYMENTS'].fillna(the_data['PAYMENTS'].median())
    the_data['MINIMUM_PAYMENTS'] = the_data['MINIMUM_PAYMENTS'].fillna(the_data['MINIMUM_PAYMENTS'].median())

    numerical = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'MINIMUM_PAYMENTS', 'CREDIT_LIMIT', 'PAYMENTS']
    categorical = ['CUST_ID']
    the_data = the_data[numerical]

    # Interpolating data
    #the_data = the_data.interpolate(method='linear')

    the_data = the_data.applymap(lambda x: np.log(x + 1))

    # Scaling data
    scalar = StandardScaler()
    data_scaled = scalar.fit_transform(the_data.values)
    data = pd.DataFrame(data_scaled, columns=the_data.columns)

    return data


def elbow_algorithm(the_data):
    print("[ ELBOW ALGORITHM ... ]")

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(4, 14))

    visualizer.fit(the_data)  # Fit the data to the visualizer
    visualizer.show()
    print("     Estimated number of clusters: "+str(visualizer.elbow_value_))
    return visualizer.elbow_value_


def exploratory_analysis(the_data, cluster_num):
    print("[ EXPLORATORY DATA ANALYSIS ... ]")
    k_means = KMeans(n_clusters=cluster_num)

    cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']
    chosen_cols = pandas.DataFrame(the_data[cols])

    label = k_means.fit_predict(chosen_cols)

    chosen_cols['CLUSTER_NUM'] = label
    cols.append('CLUSTER_NUM')

    # Graphical representation of clusters by number of elements
    sb.countplot(data=chosen_cols, hue='CLUSTER_NUM', x='CLUSTER_NUM')
    sb.pairplot(chosen_cols[cols], hue='CLUSTER_NUM', vars=cols, palette='muted')
    plt.show()

    # # BoxPlots
    ax = sb.boxplot(x="CLUSTER_NUM", y="CREDIT_LIMIT", data=chosen_cols)
    plt.title("Credit limit values by clusters ")
    plt.show()
    ax = sb.boxplot(x="CLUSTER_NUM", y="PURCHASES", data=chosen_cols)
    plt.title("Purchases values by clusters ")
    plt.show()
    ax = sb.boxplot(x="CLUSTER_NUM", y="CASH_ADVANCE", data=chosen_cols)
    plt.title("Cash advance values by clusters ")
    plt.show()
    ax = sb.boxplot(x="CLUSTER_NUM", y="BALANCE", data=chosen_cols)
    plt.title("Balance values by clusters ")
    plt.show()
    ax = sb.boxplot(x="CLUSTER_NUM", y="PAYMENTS", data=chosen_cols)
    plt.title("Payments values by clusters ")
    plt.show()
    ax = sb.boxplot(x="CLUSTER_NUM", y="MINIMUM_PAYMENTS", data=chosen_cols)
    plt.title("Minimum payments values by clusters ")
    plt.show()


def principal_component_analysis(the_data, cluster_num):

    # Reducing data dimensionality
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(the_data)

    plt.figure(figsize=(8, 6))

    plt.title('Number of clusters: '+str(cluster_num))
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    model = KMeans(n_clusters=cluster_num).fit(data_pca)
    model_label = model.labels_
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=model_label, cmap='Spectral')
    plt.colorbar(scatter)
    plt.show()


if __name__ == '__main__':
    data = load_data()
    cluster_number = elbow_algorithm(data)
    principal_component_analysis(data, cluster_number)
    exploratory_analysis(data, cluster_number)
    cluster_report(data, cluster_number)
    print("ANALYSIS DONE!")