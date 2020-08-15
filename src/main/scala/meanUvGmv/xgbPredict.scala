package meanUvGmv

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel
import org.apache.spark.sql.{Column, SparkSession}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
class xgbPredict {

}
object xgbPredict {
  def main(args: Array[String]): Unit = {
      if (args.length < 2){
        println("help: input args, log_date, model_path")
      }

    val log_date = args(0)
    val gmv_path = args(1)
    val spark = SparkSession.builder().appName("search_jixiaozhan_gmvPredict").enableHiveSupport().getOrCreate()
    import spark.implicits._
    var resFeaturesDF = spark.sql(f"select * from search_algorithm.gmv_item_profile_feature where date_id='${log_date}'")
    resFeaturesDF = resFeaturesDF.drop("date_id")

    val removeColumnNames = Seq("repurchase_rate_7d", "confirm_price_60d", "sales_productinfo_7d", "confirm_gmv_15d", "wishlist_1d", "sales_ctgr_7d",
      "shipping_sales_15d", "refund_7d", "in_sales", "buyer_female_15d", "neutral_rate_90d", "cvr_ctgr", "comment_cnt_15d", "confirm_price_unit_india_avg_15d",
      "sales_productinfo_15d", "sales_main_15d", "ctr_pv_search_15d", "sales_15d", "add_uv_30d", "confirm_gmv_30d", "confirm_male_rate_60d", "confirm_neutral_rate_15d",
      "click_uv_ctgr_1d", "confirm_male_rate_15d", "female_rate_7d", "in_sales_7d", "buyer_male_7d", "add_uv_7d", "ctr_pv_main_7d", "confirm_female_rate_90d",
      "confirm_neutral_rate_30d", "sales_productinfo_30d", "add_uv_60d", "confirm_neutral_rate_7d", "confirm_male_rate_30d", "refund_60d", "add_uv_15d",
      "female_rate_15d", "confirm_neutral_rate_90d", "confirm_price_15d", "confirm_male_rate_90d", "confirm_sales", "confirm_price_7d", "click_pv_productinfo",
      "in_confirm_sales", "add_uv_90d", "buyer_female_30d", "sales_ctgr_60d", "confirm_neutral_rate_60d", "confirm_gmv_90d", "sales_productinfo_60d",
      "confirm_price_30d", "shipping_sales", "buyer_male_15d", "buyer_female_7d", "click_uv_productinfo", "shipping_sales_1d", "buyer_female_60d", "sales_search",
      "confirm_female_rate_60d", "confirm_buyer_male", "shipping_sales_30d", "neutral_rate_1d", "good_score_rate", "ctr_uv_main_7d", "confirm_gmv_60d",
      "confirm_buyer_female", "wr_1d", "sales_ctgr_30d", "cvr_main", "buyer_neutral_7d", "sales_main_60d", "comment_cnt_60d", "orders", "confirm_buyer_female_15d",
      "buyer_female_1d", "confirm_sales_7d", "in_sales_90d", "score_size_60d", "add_uv_1d", "male_rate_1d", "click_uv_main_1d", "sales_productinfo_1d", "buyer_female_90d",
      "buyers", "shipping_sales_90d", "buyer_male_1d", "confirm_price_unit_india_var_1d", "buyer_male_60d", "confirm_female_rate_30d", "buyer_male_30d",
      "click_uv_productinfo_1d", "shipping_sales_60d", "click_pv_main", "buyer_neutral_30d", "confirm_buyer_neutral", "sales_ctgr_90d", "click_uv_main",
      "confirm_buyer_male_15d", "buyer_neutral_15d", "score_60d", "buyer_male_90d", "score_size_90d", "confirm_female_rate_15d", "confirm_buyer_female_30d",
      "buyer_neutral_1d", "confirm_female_rate_7d", "confirm_buyer_male_30d", "confirm_male_rate_7d", "shipping_orders_7d", "shipping_orders_30d", "confirm_orders",
      "comment_cnt_1d", "score_description_60d", "shipping_orders_60d", "shipping_orders_15d", "sales_productinfo_90d", "score_90d", "refund_rate_1d", "confirm_buyer_neutral_7d",
      "score_quality_60d", "in_confirm_sales_7d", "confirm_price_1d", "orders_7d", "score_quality_90d", "in_buyers", "confirm_buyer_male_7d", "in_sales_1d", "score_size_30d",
      "confirm_buyer_female_7d", "score_description_90d", "refund_90d", "in_sales_60d", "sales_ctgr", "buyers_7d", "score_quality_30d", "confirm_sales_30d",
      "shipping_orders", "orders_15d", "female_rate_1d", "confirm_sales_60d", "confirm_neutral_rate_1d", "confirm_buyer_female_60d", "sales_main_1d",
      "buyer_neutral_60d", "score_quality_15d", "in_sales_30d", "confirm_sales_15d", "score_description_30d", "buyers_15d", "refund_1d", "confirm_buyer_neutral_15d",
      "ctr_pv_productinfo_7d", "buyers_90d", "buyer_neutral_90d", "confirm_male_rate_1d", "score_30d", "shipping_orders_90d", "sales_main_90d", "score_description_7d",
      "score_size_15d", "confirm_buyer_neutral_30d", "score_description_15d", "in_confirm_sales_30d", "buyers_60d", "in_confirm_buyers", "confirm_buyer_male_90d",
      "confirm_buyer_male_1d", "score_size_7d", "confirm_sales_1d", "confirm_buyer_neutral_60d", "in_confirm_sales_60d", "sales_productinfo", "confirm_sales_90d",
      "in_sales_15d", "orders_30d", "score_quality_7d", "sales_ctgr_1d", "confirm_buyer_neutral_90d", "buyers_1d", "confirm_buyer_male_60d", "orders_90d",
      "good_score_rate_90d", "confirm_buyers", "confirm_buyer_female_1d", "orders_1d", "confirm_orders_15d", "orders_60d", "confirm_buyer_neutral_1d",
      "score_size_1d", "shipping_orders_1d", "sales_main", "good_score_rate_30d", "in_buyers_7d", "good_score_rate_60d", "score_7d", "confirm_orders_90d",
      "buyers_30d", "in_buyers_30d", "confirm_buyers_7d", "in_confirm_sales_15d", "score_15d", "confirm_buyer_female_90d", "score_quality_1d", "confirm_buyers_15d",
      "in_buyers_15d", "confirm_buyers_30d", "in_confirm_sales_90d", "confirm_orders_60d", "in_confirm_buyers_30d", "confirm_orders_7d", "good_score_rate_7d",
      "in_buyers_90d", "confirm_buyers_90d", "in_buyers_1d", "in_confirm_buyers_60d", "confirm_female_rate_1d", "score_description_1d", "in_confirm_buyers_15d",
      "confirm_buyers_1d", "good_score_rate_15d", "confirm_buyers_60d", "in_buyers_60d", "score_1d", "in_confirm_sales_1d", "in_confirm_buyers_90d",
      "confirm_orders_1d", "confirm_orders_30d", "in_confirm_buyers_7d", "good_score_rate_1d", "in_confirm_buyers_1d")

    resFeaturesDF = resFeaturesDF.drop(removeColumnNames:_*)
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

    val xgbRegressionModel = XGBoostRegressionModel.load(gmv_path + "/model")
    var predict_results = xgbRegressionModel.transform(xgbInput)
    predict_results = predict_results.withColumnRenamed("prediction", "offlineAvgUvGmvScore_v1")
    predict_results.select($"pid", $"offlineAvgUvGmvScore_v1").write.mode("overwrite").option("header", "true").csv(gmv_path + "/predict_results/" +  log_date)

  }
}
