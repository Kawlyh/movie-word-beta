# movie-word-beta
使用IMDB电影评论数据进行情感分析
# 结构：
1. wash.py：分词清洗
2. process-word2vec：使用word2vec得到单词特征向量
3. sentence.py：将段落裁开为句子列表
4. makefeature.py：得到平均特征向量
5. process-ave-vec：在234的基础上，对所有评论进行向量平均，再使用RandomForest进行测试数据预测
