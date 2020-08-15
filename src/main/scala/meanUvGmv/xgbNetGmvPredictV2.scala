package meanUvGmv

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
class xgbNetGmvPredictV2 {

}
object xgbNetGmvPredictV2 {
  def main(args: Array[String]): Unit = {
    if (args.length < 2){
      println("help: input args, log_date, model_path")
    }

    val log_date = args(0)
    val gmv_path = args(1)
    val spark = SparkSession.builder().appName("search_jixiaozhan_netGmvPredictV1").enableHiveSupport().getOrCreate()
    import spark.implicits._
    var resFeaturesDF = spark.sql(f"select * from search_algorithm.reversed_item_profile_v2 where date_id='${log_date}'")
    resFeaturesDF = resFeaturesDF.drop("date_id")
    resFeaturesDF = resFeaturesDF.na.fill(0)
    resFeaturesDF = resFeaturesDF.na.drop()

    def func(column: Column) = column.cast(DoubleType)
    val itemColumnName = resFeaturesDF.columns
    val itemFeaturesDF = resFeaturesDF.select(itemColumnName.map(name => func(col(name))): _*)

    val featureColumnName = itemColumnName.drop(1)
    val vectorAssembler = new VectorAssembler().setInputCols(featureColumnName).setOutputCol("features")

    val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)

    val xgbInput = vectorAssembler.transform(itemFeaturesDF).select($"pid", toDense($"features").alias("features"))
    //    xgbInput.foreach { row =>
    //      row.toSeq.foreach{col => print(col) }
    //    }
    xgbInput.repartition(256).persist(StorageLevel.MEMORY_AND_DISK)

    val xgbRegressionModel = XGBoostRegressionModel.load(gmv_path + "/net_model_v2")
    var predict_results = xgbRegressionModel.transform(xgbInput)
    predict_results = predict_results.withColumnRenamed("prediction", "offlineAvgUvGmvScore_v1")
    predict_results.select($"pid", $"offlineAvgUvGmvScore_v1").write.mode("overwrite").option("header", "true").csv(gmv_path + "/netGmvV2_predict_results/" +  log_date)

  }
}
