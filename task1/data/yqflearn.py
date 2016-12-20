#!/usr/bin/python

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import operator

class KNN(BaseEstimator, ClassifierMixin):

    def __init__(self, dis_func='euclidean', scale=True, K=1):
        """
        dis_func 距离函数，可取DistanceMetric中可用的距离函数名称
        scale 是否对输入的特征向量进行均一化处理
        K 使用多少个近邻
        """
        self.dis_func = DistanceMetric.get_metric(dis_func)
        self.scale = scale
        self.K = K
        self.encoder = LabelEncoder()

    def select_class(self, neighbours):
        """
        选取近邻中类别最多的一类作为该样本的类别
        """
        votes = [0] * len(self.classes_)
        classes = self.encoder.transform(self.y_train[neighbours])
        for id in classes:
            votes[id] += 1
        return self.encoder.inverse_transform(np.argmax(votes))

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.encoder.fit(y)

        if self.scale:
            self.scaler = MinMaxScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)

        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):
        check_is_fitted(self, ['X_train', 'y_train', 'scale', 'K'])

        X = check_array(X)

        if self.scale:
            X = self.scaler.transform(X)

        closest = None

        # 为了节省内存和加快运算速度
        # 每1000个样本批处理
        STEP = 1000
        k_enabled = self.K != 1 and self.K < len(X)
        for start in range(0, len(X), STEP):
            x = X[start:start+STEP]
            distances = self.dis_func.pairwise(x, self.X_train)
            self.distances = distances
            if k_enabled:
                k_closest = np.argpartition(distances, self.K, axis=1)[:, 0:self.K]
                closest_chunck = list(map(self.select_class, k_closest))
            else:
                closest_chunck = np.argmin(distances, axis=1)
            if closest is None:
                closest = closest_chunck
            else:
                closest = np.append(closest, closest_chunck)
        
        if k_enabled:
            return closest
        else:
            return self.y_train[closest]

    def get_params(self, deep=True):
        return {"dis_func": self.dis_func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

class NativeBayes(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        self.X_train = X
        self.y_train = y

        self.encoder.fit(y)

        prior_probabilities = [0.0] * len(self.classes_)
        self.prior_probabilities = prior_probabilities

        dimension_data = [None] * len(self.classes_)
        self.dimension_data = dimension_data

        # 计算先验概率等统计信息
        total_instances_num = len(y)
        dimension_num = X.shape[1]
        for class_name, class_id in zip(self.classes_, self.encoder.transform(self.classes_)):
            instances = X[y == class_name]
            prior_probabilities[class_id] = len(instances) / total_instances_num
            dimension_info = [None] * dimension_num
            for dim in range(dimension_num):
                dim_data = instances[:, dim]
                mean = np.mean(dim_data)
                var = np.var(dim_data, ddof=1)
                dimension_info[dim] = (mean, var)
            dimension_data[class_id] = dimension_info

        return self

    def predict(self, X):

        check_is_fitted(self, ['dimension_data', 'prior_probabilities'])

        X = check_array(X)
        pred = np.zeros(len(X))

        # 利用先验概率和统计信息
        # 计算每个样本属于每一类的概率，选取概率最大的类标
        for ins_id, instance in enumerate(X):
            post_probabilities = [0.0] * len(self.classes_)
            for idx, class_data in enumerate(self.dimension_data):
                post_probabilities[idx] = self.prior_probabilities[idx]
                for dim_idx, dimension_info in enumerate(class_data):
                    data = instance[dim_idx]
                    mean = dimension_info[0]
                    var = dimension_info[1]

                    p1 = 1.0 / (np.sqrt(2 * np.pi * var))
                    p2 = np.exp(( -1 * pow((data - mean), 2)) / (2.0 * var))

                    if np.isinf(np.abs(p1)) or np.isinf(np.abs(p2)):
                        continue
                    if np.isnan(p1) or np.isnan(p2):
                        continue

                    post_probabilities[idx] *= p1 * p2

            post_probabilities = post_probabilities / np.sum(post_probabilities)
            pred[ins_id] = self.encoder.inverse_transform(np.argmax(post_probabilities))
        return pred

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self
