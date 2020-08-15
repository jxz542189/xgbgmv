package sales7d

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.storage.StorageLevel

class Sales7dXgbPredict {

}

object Sales7dXgbPredict {
  def main(args: Array[String]): Unit = {
    if (args.length < 2){
      println("input args: log_date, model_path")
    }

    val log_date = args(0)
    val model_dir = args(1)
    println(s"Sales7dXgbPredict - log_date: $log_date, model_dir: $model_dir")

    val spark = SparkSession.builder().appName("search_dupeng_salesPredict").enableHiveSupport().getOrCreate()
    import spark.implicits._

    // 读取feature数据（过滤掉不在架的商品）, [pid, feature1, feature2, ..., date_id]
    val featuresDF = spark.sql(f"select * from search_algorithm.gmv_item_profile_feature where is_sale = 1 and date_id='$log_date'")
    println(s"$log_date featuresDF count: ${featuresDF.count()}")

    // feature处理
    // 1. 转Double
    val tempColNames = Array("pid", "date_id")
    def castDouble(colName: String) = {
      if (tempColNames.contains(colName)) {
        col(colName)
      } else {
        col(colName).cast(DoubleType)
      }
    }
    val itemColNames = featuresDF.columns
    var itemDF = featuresDF.select(itemColNames.map(name => castDouble(name)): _*)

    // 2. VectorAssembler
    val featureColNames = itemColNames.drop(1).dropRight(1)  // drop: pid | date_id, result: [feature1, feature2, ...]
    println(s"features count: ${featureColNames.length}")
    val vectorAssembler = new VectorAssembler().setInputCols(featureColNames).setOutputCol("features")
    itemDF = vectorAssembler.transform(itemDF)  // [pid, features, date_id]

    // 3. toDense
    val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)
    // [pid, date_id, features]
    val xgbInput = itemDF.select($"pid", $"date_id", toDense($"features").alias("features"))
    println(s"xgbInput: \n ${xgbInput.orderBy(rand).show(10)}")

    xgbInput.persist(StorageLevel.MEMORY_AND_DISK)

    // 加载模型
    println("load model...")
    val xgbRegressionModel = XGBoostRegressionModel.load(model_dir + "/model")

    // 模型预测，[pid, date_id, features, prediction]
    println("predict...")
    var predict_results = xgbRegressionModel.transform(xgbInput)

    // 将原始预测值还原成销量，[pid, date_id, features, prediction, prediction_recovered]
    predict_results = predict_results.withColumn("prediction_recovered", expm1("prediction"))
    println(s"prediction results: \n ${predict_results.orderBy(rand).show(100)}")

    // 保存预测结果，注意：这里不能用coalesce(resPartitions)，否则会导致数据丢失和错乱！[pid, date_id, prediction, prediction_recovered]
    val savePath = model_dir + "/predict_results/" + log_date
    println(s"save results to $savePath...")
    predict_results.drop("features").write.mode("overwrite").option("header", "true").csv(savePath)
    println("done")
  }
}
