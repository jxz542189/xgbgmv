package sales7d

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.storage.StorageLevel

object Sales7dXgbValidate {
    def main(args: Array[String]): Unit = {
        // 加载模型重新在测试集上预测，验证结果是否一致

        val spark = SparkSession.builder().appName("search_dupeng_sales7d_val").enableHiveSupport().getOrCreate()
        import spark.implicits._

        val base_dir = "s3://jiayun.spark.data/product_algorithm/offline_quality_dev/sales_7d_log1p_lr0.2_d10_round1000_6days_v2"

        // 获取全量数据的features, [pid, feature1, feature2, ..., date_id]
        var featuresDF = spark.sql(f"select * from search_algorithm.gmv_item_profile_feature where is_sale = 1 and date_id in ('2019-11-05', '2019-11-06', '2019-11-07', '2019-11-08', '2019-11-09', '2019-11-10')")
        println(s"featuresDF count: ${featuresDF.count()}")

        // 读取test_result, [pid, date_id, label, label_converted, prediction, prediction_recovered]
        val numPartitions = 1200
        var df_test_result = spark.emptyDataFrame
        for (i <- 0 until numPartitions) {
            val seq = "%05d".format(i)
            val file = s"/test_result/part-$seq-0782199f-a5db-4e1b-9d43-4392a5b6ab93-c000.csv"
            val df_sub = spark.read.format("csv")
              .option("inferSchema", "true")
              .option("header", "true")
              .load(base_dir + file)

            if (df_test_result == spark.emptyDataFrame) {
                df_test_result = df_sub
            } else {
                df_test_result = df_test_result.union(df_sub)
            }
        }
        println(s"df_test_result count: ${df_test_result.count()}")

        // 通过join过滤，只保留test数据及其features,  [pid, date_id, feature1,...]
        val df_test_key = df_test_result.select("pid", "date_id")
        featuresDF = featuresDF.join(df_test_key, Seq("pid", "date_id"), "inner")
        println(s"after filtered, featuresDF count:${featuresDF.count()}")

        // feature处理
        println("process features...")

        // 1. 转Double
        val tempColNames = Array("pid", "date_id")
        def func(colName: String) = {
            if (tempColNames.contains(colName)) {
                col(colName)
            } else {
                col(colName).cast(DoubleType)
            }
        }
        val itemColNames = featuresDF.columns
        var itemDF = featuresDF.select(itemColNames.map(name => func(name)): _*)

        // 2. VectorAssembler
        val featureColNames = itemColNames.drop(2)  // drop: [pid, date_id], result: [feature1, feature2, ...]
        println(s"features count: ${featureColNames.length}")
        val vectorAssembler = new VectorAssembler().setInputCols(featureColNames).setOutputCol("features")
        itemDF = vectorAssembler.transform(itemDF)  // [pid, features, date_id]

        // 3. toDense
        val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)
        val xgbInput = itemDF.select($"pid", $"date_id", toDense($"features").alias("features"))

        xgbInput.persist(StorageLevel.MEMORY_AND_DISK)

        // 加载模型
        println("load model...")
        val xgbRegressionModel = XGBoostRegressionModel.load(base_dir + "/model")

        // 模型预测
        println("predict...")
        var predict_results = xgbRegressionModel.transform(xgbInput)

        // 还原预测值, [pid, date_id, features, model_prediction, model_recovered_prediction]
        predict_results = predict_results
          .withColumn("model_prediction_recovered", expm1("prediction"))
          .withColumnRenamed("prediction", "model_prediction")

        // 和原始的测试结果数据join，以便对比验证,
        // [pid, date_id, label, label_converted, prediction, prediction_recovered, model_prediction, model_recovered_prediction]
        val df_merge = df_test_result.join(predict_results.drop("features"), Seq("pid", "date_id"), "left")

        // 保存预测结果
        println("save prediction...")
        df_merge
          .write
          .mode("overwrite")
          .option("header", "true")
          .csv(base_dir + "/test_result_predict")
    }
}
