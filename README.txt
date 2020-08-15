# 打包编译
	 mvn clean scala:compile compile package

# UV价值预测模型训练
%sh
spark-submit \
--class meanUvGmv.xgbTrain \
--master yarn \
--deploy-mode client \
--executor-memory 20g \
--name search_jixiaozhan_meanUvGmv \
--queue root.sparkOffline \
s3://jiayun.spark.data/product_algorithm/mean_uv_gmv/xgb-1.0-SNAPSHOT.jar \
10000 2019-11-05,2019-11-06 2019-11-06,2019-11-07 0.02 10 10 1 \
s3://jiayun.spark.data/product_algorithm/mean_uv_gmv


# 销量预测模型训练
%sh
spark-submit \
--class sales7d.Sales7dXgbTrain \
--master yarn \
--deploy-mode client \
--name search_dupeng_sales \
--queue root.sparkOffline \
s3://jiayun.spark.data/product_algorithm/offline_quality_dev/sales_7d/models/s001_lr0.15_round1600_d10_early20_v1/xgb-1.0-SNAPSHOT.jar \
s3://jiayun.spark.data/product_algorithm/offline_quality_dev/sales_7d/samples/6days_sr0.01_mean_lop1p_001 \
0.15 1600 10 20 90 rmse true \
s3://jiayun.spark.data/product_algorithm/offline_quality_dev/sales_7d/models/s001_lr0.15_round1600_d10_early20_v1 WARN


%sh
spark-submit \
--master yarn \
--deploy-mode client \
--executor-memory 2g \
--name search_jixiaozhan_meanUvGmv \
--queue root.sparkOffline \
--jars s3://jiayun.spark.data/search-center/liujianshi/complete_personalize/spark-tensorflow-connector_2.11-1.13.1.jar \
s3://jiayun.spark.data/product_algorithm/query_category/ctr_model/tfrecord.py


