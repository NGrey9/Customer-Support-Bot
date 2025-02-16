import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent import Agent
load_dotenv()

AGENT_RECOMMENDATION = os.environ['AGENT_RECOMMENDATION']


class RecommendationAgent(Agent):

    prompt = """
    <prompt>
    """
    model_name = AGENT_RECOMMENDATION

    def __init__(self, sql_database_api):
        super().__init__(agent_name="RecommendationAgent")
        self.sql_database_api = sql_database_api
        self.create_interaction_matrix()
        self.create_prediction_matrix(k=9)
        self.create_product_index_mapping()

    def create_prediction_matrix(self, k: int):
        R = self.user_item_matrix.values

        u, s, vt = svds(R, k=k)

        s_diag_matrix = np.diag(s)

        self.R_hat = np.dot(np.dot(u, s_diag_matrix), vt)

    def create_interaction_matrix(self):
        orders_df = pd.read_sql("SELECT * FROM orders",
                                self.sql_database_api.engine)
        order_items_df = pd.read_sql(
            "SELECT * FROM order_items", self.sql_database_api.engine)

        order_items_joined = pd.merge(order_items_df,
                                      orders_df[['id', 'user_id']],
                                      left_on='order_id', right_on='id',
                                      suffixes=('', '_order'))

        order_items_joined = order_items_joined[[
            'user_id', 'product_id', 'quantity']]

        order_items_joined['interaction'] = 1

        self.user_item_matrix = order_items_joined.pivot_table(index='user_id',
                                                               columns='product_id',
                                                               values='interaction',
                                                               aggfunc='max',
                                                               fill_value=0)

    def create_product_index_mapping(self):
        products_df = pd.read_sql(
            "SELECT * FROM products", self.sql_database_api.engine)
        products_df['text'] = products_df['name'].fillna(
            '') + " " + products_df['description'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(products_df['text'])
        self.product_index_mapping = pd.Series(
            products_df.index, index=products_df['id']).to_dict()

    def get_user_profile(self, user_id):
        purchased_products = self.user_item_matrix.columns[self.user_item_matrix.loc[user_id] > 0]

        indices = [self.product_index_mapping[pid]
                   for pid in purchased_products if pid in self.product_index_mapping]
        if not indices:
            return None
        profile_vector = self.tfidf_matrix[indices].mean(axis=0)
        return profile_vector

    def get_cb_scores(self, user_id):
        user_profile = self.get_user_profile(user_id)
        if user_profile is None:
            # Nếu user chưa mua sản phẩm nào, trả về mảng zeros
            return np.zeros(self.tfidf_matrix.shape[0])
        # Tính cosine similarity: kết quả là mảng (1, n_products)
        user_profile_array = np.asarray(user_profile)
        tfidf_matrix_array = self.tfidf_matrix.toarray()
        scores = cosine_similarity(user_profile_array, tfidf_matrix_array)
        return scores.flatten()

    def hybrid_recommend_product(self, user_id, alpha=0.5):

        if user_id not in self.user_item_matrix.index:
            return None

        # Lấy danh sách product_ids theo thứ tự cột trong user_item_matrix (cũng là thứ tự của ma trận CF)
        cf_product_ids = self.user_item_matrix.columns

        # CF scores: lấy từ ma trận R_hat
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        cf_scores = self.R_hat[user_idx]

        # CBF scores: lấy theo thứ tự sản phẩm trong products_df, sau đó ánh xạ lại theo cf_product_ids
        # độ dài = số lượng sản phẩm trong products_df
        cb_scores_full = self.get_cb_scores(user_id)
        cb_scores = []
        for pid in cf_product_ids:
            # Nếu product_id có mapping, lấy điểm CB; nếu không, gán 0
            if pid in self.product_index_mapping:
                idx = self.product_index_mapping[pid]
                cb_scores.append(cb_scores_full[idx])
            else:
                cb_scores.append(0)
        cb_scores = np.array(cb_scores)

        # Hàm normalize: quy chuẩn điểm về khoảng [0, 1]
        def normalize(arr):
            if arr.max() == arr.min():
                return np.zeros_like(arr)
            return (arr - arr.min()) / (arr.max() - arr.min())

        cf_norm = normalize(cf_scores)
        cb_norm = normalize(cb_scores)

        # Tính điểm cuối cùng theo trọng số alpha
        final_scores = alpha * cf_norm + (1 - alpha) * cb_norm

        # Loại trừ các sản phẩm mà user đã mua
        purchased = set(
            self.user_item_matrix.columns[self.user_item_matrix.loc[user_id] > 0])

        # Sắp xếp các sản phẩm theo final_scores giảm dần và chọn sản phẩm đầu tiên chưa mua
        sorted_indices = np.argsort(-final_scores)
        for idx in sorted_indices:
            product_id = cf_product_ids[idx]
            if product_id not in purchased:
                return product_id

        return None
