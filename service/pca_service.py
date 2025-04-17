import re

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

import ast  # 문자열을 실제 리스트로 변환하기 위한 라이브러리

class PCAService:
    def __init__(self):
        self.scaler = StandardScaler()
        # self.df = pd.read_csv('data/openAI_embedding_Test.csv')
        self.df = pd.read_csv('data/embed_test_data.csv')

    def __parse_embedding(self, embedding_str):
        cleaned_str = embedding_str.strip().strip('[]')
        cleaned_str = cleaned_str.replace(",","")
        return np.array([float(num) for num in cleaned_str.split()])


    def visualize(self):
        # AO 컬럼을 실제 인덱스로 바꿀 것 (AO는 0부터 시작하면 40번째임)
        embedding_index = 40  # AO번째가 실제로는 40번째(0부터 세었을 때)
        embeddings = self.df.iloc[:, embedding_index].apply(self.__parse_embedding)
        embeddings_matrix = np.vstack(embeddings.values)  # 2D 배열로 변환

        # 3. PCA로 차원 축소 (1536차원 → 2차원으로 축소)
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_matrix)

        # 4. 시각화하기
        plt.figure(figsize=(10, 7))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.6, s=50)
        plt.title('PCA Visualization of Embeddings')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True)
        plt.show()

    def visualize_3d(self):
        embedding_index = 40
        embeddings = self.df.iloc[:, embedding_index].apply(self.__parse_embedding)
        embeddings_matrix = np.vstack(embeddings.values)
        pca = PCA(n_components=3)
        reduced_embeddings = pca.fit_transform(embeddings_matrix)

        original_features = self.df.select_dtypes(include=[np.number])
        pca_corr = pd.DataFrame(reduced_embeddings, columns=['PC1', 'PC2', 'PC3']).join(original_features).corr()
        print(pca_corr[['PC1', 'PC2', 'PC3']].sort_values(by='PC1', ascending=False))
        # PCA 결과 DataFrame 생성 (원본 데이터 정보 포함)
        result_df = pd.DataFrame({
            'PC1': reduced_embeddings[:, 0],
            'PC2': reduced_embeddings[:, 1],
            'PC3': reduced_embeddings[:, 2],
            'original_info': self.df.iloc[:, 1],  # 예시로 첫 번째 컬럼 사용
            'upjong': self.df.iloc[:, 6]
        })

        # PCA가 설명하는 분산 비율 출력 (각 PC의 중요성 파악)
        explained_variance = pca.explained_variance_ratio_
        print(f'PC1이 설명하는 비율: {explained_variance[0]:.2%}')
        print(f'PC2가 설명하는 비율: {explained_variance[1]:.2%}')
        print(f'PC3가 설명하는 비율: {explained_variance[2]:.2%}')
        print(f'총 설명된 분산 비율: {explained_variance.sum():.2%}')

        # 3D Plotly로 interactive 시각화
        fig = px.scatter_3d(
            result_df, x='PC1', y='PC2', z='PC3',
            color='upjong',
            hover_data=['original_info'],
            title='3D PCA Visualization of Embeddings'
        )

        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.update_layout(height=700, width=800)

        fig.show()