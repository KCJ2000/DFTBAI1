import torch
import numpy as np
import time

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        初始化 KMeans 类
        :param n_clusters: 聚类中心的数量
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        :param device: 运行设备，默认为 GPU，如果无 GPU 则使用 CPU
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.centroids = None

    def fit(self, X):
        """
        对数据进行 K-means 聚类
        :param X: 输入数据，形状为 (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        # 随机初始化聚类中心
        self.centroids = X[torch.randint(0, n_samples, (self.n_clusters,))].to(self.device)

        for _ in range(self.max_iter):
            # 计算每个点到聚类中心的距离
            dist_matrix = self._compute_distances(X)
            # 分配每个点到最近的聚类中心
            labels = torch.argmin(dist_matrix, dim=1)
            # 更新聚类中心
            new_centroids = self._update_centroids(X, labels)

            # 检查是否收敛
            if torch.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        return labels

    def _compute_distances(self, X):
        """
        计算每个点到聚类中心的距离
        :param X: 输入数据，形状为 (n_samples, n_features)
        :return: 距离矩阵，形状为 (n_samples, n_clusters)
        """
        X = X.to(self.device)
        dist_matrix = torch.cdist(X, self.centroids, p=2.0)
        return dist_matrix

    def _update_centroids(self, X, labels):
        """
        更新聚类中心
        :param X: 输入数据，形状为 (n_samples, n_features)
        :param labels: 每个点的聚类标签，形状为 (n_samples,)
        :return: 新的聚类中心，形状为 (n_clusters, n_features)
        """
        X = X.to(self.device)
        labels = labels.to(self.device)
        new_centroids = torch.zeros_like(self.centroids)

        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if cluster_points.shape[0] > 0:
                new_centroids[i] = torch.mean(cluster_points, dim=0)

        return new_centroids

# 示例用法
if __name__ == "__main__":
    # 生成随机 3 维数据
    np.random.seed(0)
    torch.manual_seed(0)
    data_size = 1000
    dims = 3
    data = np.random.randn(data_size, dims) / 6
    data = torch.as_tensor(data,dtype=torch.float32)
    print(data)
    
    # 设置聚类参数
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # 运行 K-means 算法
    start_time = time.time()
    labels = kmeans.fit(data)
    end_time = time.time()

    print(f"K-means 聚类完成，耗时 {end_time - start_time:.4f} 秒")
    print(f"聚类结果（部分）：{labels[:100]}")