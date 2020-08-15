package sales7d

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

object Sales7dXgbResult {
  def main(args: Array[String]): Unit = {
    // 计算模型训练、测试、验证结果的rmse指标

    val spark = SparkSession.builder().appName("search_dupeng_sales7d_result").enableHiveSupport().getOrCreate()
    import spark.implicits._

    val base_dir = "s3://jiayun.spark.data/product_algorithm/offline_quality_dev/sales_7d_log1p_lr0.15_d10_round1400_6days_v1"

    // 读取test_result, [pid, date_id, label, label_converted, prediction, prediction_recovered]
    val numPartitions = 1200
    var df_result = spark.emptyDataFrame
    for (i <- 0 until numPartitions) {
      val seq = "%05d".format(i)
      val file = s"/test_result/part-$seq-4be3e4f4-e670-4d13-9ed0-8a64192632d1-c000.csv"
      val df_sub = spark
        .read
        .format("csv")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(base_dir + file)

      if (df_result == spark.emptyDataFrame) {
        df_result = df_sub
      } else {
        df_result = df_result.union(df_sub)
      }
    }
    println(s"df_result count: ${df_result.count()}")

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction_recovered").setMetricName("rmse")
    val rmse = evaluator.evaluate(df_result)
    println(s"rmse: $rmse")
  }

}
