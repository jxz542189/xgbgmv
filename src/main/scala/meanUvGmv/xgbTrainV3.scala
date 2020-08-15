package meanUvGmv

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, rand, udf}
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

class xgbTrainV3{

}


object xgbTrainV3{
  def main(args: Array[String]): Unit = {
    if (args.length < 8){
      println("help: input args, numRound, log_dates, label_log_dates, sample_rate, nTreeDepth, numWorkers, nThread, xgbModelPath")
    }

    val numRound = args(0).toInt
    val log_dates = args(1).split(',')
    val label_log_dates = args(2).split(',')
    val sample_rate = args(3).toFloat
    val nTreeDepth = args(4).toInt
    val numWorkers = args(5).toInt
    val eta = args(6).toFloat
    val xgbModelPath = args(7)
    val spark = SparkSession.builder().appName("search_jixiaozhan_gmvTrainV2").enableHiveSupport().getOrCreate()
    //    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._
    val removeColumnNames = Seq("ctr_uv_1d", "acr_uv_1d", "wr_uv_1d", "ctr_uv_7d", "acr_uv_7d", "wr_uv_7d",
      "ctr_uv_15d", "acr_uv_15d", "wr_uv_15d", "ctr_uv_30d", "acr_uv_30d", "wr_uv_30d",
      "ctr_uv_60d", "acr_uv_60d", "wr_uv_60d", "ctr_uv_90d", "acr_uv_90d", "wr_uv_90d",
      "ctr_1d", "acr_1d", "wr_1d", "ctr_7d", "acr_7d", "wr_7d",
      "ctr_15d", "acr_15d", "wr_15d", "ctr_30d", "acr_30d", "wr_30d",
      "ctr_60d", "acr_60d", "wr_60d", "ctr_90d", "acr_90d", "wr_90d",
      "ctr_uv", "acr_uv", "wr_uv", "ctr", "acr", "wr",
      "ctr_pv_main", "ctr_uv_main", "ctr_pv_productinfo", "ctr_uv_productinfo",
      "ctr_pv_ctgr", "ctr_uv_ctgr", "ctr_pv_search", "ctr_uv_search",
      "ctr_pv_main", "ctr_uv_main", "ctr_pv_productinfo", "ctr_uv_productinfo",
      "ctr_pv_ctgr", "ctr_uv_ctgr", "ctr_pv_search", "ctr_uv_search",
      "ctr_pv_main_1d", "ctr_uv_main_1d", "ctr_pv_productinfo_1d", "ctr_uv_productinfo_1d",
      "ctr_pv_ctgr_1d", "ctr_uv_ctgr_1d", "ctr_pv_search_1d", "ctr_uv_search_1d",
      "ctr_pv_main_7d", "ctr_uv_main_7d", "ctr_pv_productinfo_7d", "ctr_uv_productinfo_7d",
      "ctr_pv_ctgr_7d", "ctr_uv_ctgr_7d", "ctr_pv_search_7d", "ctr_uv_search_7d",
      "ctr_pv_main_15d", "ctr_uv_main_15d", "ctr_pv_productinfo_15d", "ctr_uv_productinfo_15d",
      "ctr_pv_ctgr_15d", "ctr_uv_ctgr_15d", "ctr_pv_search_15d", "ctr_uv_search_15d",
      "ctr_pv_main_30d", "ctr_uv_main_30d", "ctr_pv_productinfo_30d", "ctr_uv_productinfo_30d",
      "ctr_pv_ctgr_30d", "ctr_uv_ctgr_30d", "ctr_pv_search_30d", "ctr_uv_search_30d",
      "ctr_pv_main_60d", "ctr_uv_main_60d", "ctr_pv_productinfo_60d", "ctr_uv_productinfo_60d",
      "ctr_pv_ctgr_60d", "ctr_uv_ctgr_60d", "ctr_pv_search_60d", "ctr_uv_search_60d",
      "ctr_pv_main_90d", "ctr_uv_main_90d", "ctr_pv_productinfo_90d", "ctr_uv_productinfo_90d",
      "ctr_pv_ctgr_90d", "ctr_uv_ctgr_90d", "ctr_pv_search_90d", "ctr_uv_search_90d")
    var log_date = log_dates(0)
    var label_log_date = label_log_dates(0)
    var resFeaturesDF = spark.sql(f"select * from search_algorithm.gmv_item_profile_feature where date_id='${log_date}'")
    resFeaturesDF = resFeaturesDF.drop(removeColumnNames : _*)
    var resReveredDF = spark.sql(f"select * from search_algorithm.offlineReversed where date_id='${log_date}'")
    resFeaturesDF = resFeaturesDF.join(resReveredDF, Seq("pid"), "inner")
    resFeaturesDF = resFeaturesDF.where("is_sale == 1")
    resFeaturesDF = resFeaturesDF.drop("date_id")
    var resLabelDF = spark.sql(f"select * from search_algorithm.offlineAvgGmv_V2 where date_id='${label_log_date}'")
    resLabelDF = resLabelDF.drop("date_id")
    resLabelDF = resLabelDF.where("search_total_uv > 0")
    resLabelDF = resLabelDF.drop("search_total_uv")
    val resLabelDF_1  = resLabelDF.where("search_mean_uv_gmv == 0").sample(sample_rate)
    val resLabelDF_2 = resLabelDF.where("search_mean_uv_gmv > 0")
    resLabelDF = resLabelDF_1.union(resLabelDF_2)
    resLabelDF = resLabelDF.drop("search_mean_uv_gmv")
    var resDF = resFeaturesDF.join(resLabelDF, Seq("pid"), "inner")
    //    println(resDF.show(1))

    for (i <- 1 to (log_dates.length - 1)){
      log_date = log_dates(i)
      var featuresDF = spark.sql(f"select * from search_algorithm.gmv_item_profile_feature where date_id='${log_date}'")
      featuresDF = featuresDF.drop(removeColumnNames : _*)
      var resReveredDF_1 = spark.sql(f"select * from search_algorithm.offlineReversed where date_id='${log_date}'")
      featuresDF = featuresDF.join(resReveredDF_1, Seq("pid"), "inner")
      featuresDF = featuresDF.where("is_sale == 1")
      featuresDF = featuresDF.drop("date_id")

      label_log_date = label_log_dates(i)
      var resLabelDF = spark.sql(f"select * from search_algorithm.offlineAvgGmv_V2 where date_id='${label_log_date}'")
      resLabelDF = resLabelDF.drop("date_id")
      resLabelDF = resLabelDF.where("search_total_uv > 0")
      resLabelDF = resLabelDF.drop("search_total_uv")
      val resLabelDF_1  = resLabelDF.where("search_mean_uv_gmv == 0").sample(sample_rate)
      val resLabelDF_2 = resLabelDF.where("search_mean_uv_gmv > 0")
      resLabelDF = resLabelDF_1.union(resLabelDF_2)
      resLabelDF = resLabelDF.drop("search_mean_uv_gmv")
      val resDF1 = featuresDF.join(resLabelDF, Seq("pid"), "inner")
      //      println(resDF1.)
      resDF = resDF.union(resDF1)
    }

    val remainColumns = Seq("cvr_pv_productinfo_1d", "add_uv_15d", "shipping_sales_60d", "repurchase_rate_15d",
      "score", "confirm_buyer_female", "score_description_60d", "refund_60d", "gmv_1d", "shipping_sales_7d",
      "add_uv_7d", "buyers", "score_description_90d", "cvr_pv_ctgr_1d", "refund_90d", "score_90d", "score_size_60d",
      "score_size_90d", "refund_30d", "cvr_uv_ctgr_1d", "buyer_male_60d", "score_description_30d", "comment_cnt_7d",
      "shipping_sales_90d", "confirm_buyer_neutral", "orders", "click_uv_productinfo_1d", "score_quality_60d",
      "in_confirm_sales", "buyer_female_30d", "buyer_male_90d", "score_quality_90d", "buyer_male_30d", "comment_cnt_15d",
      "confirm_price_unit_india_avg_1d", "buyer_female_7d", "score_size_30d", "score_description_7d", "buyer_male_15d",
      "score_size_15d", "neutral_rate_1d", "sales_productinfo_30d", "click_uv_ctgr_1d", "comment_cnt_30d", "buyer_neutral_7d",
      "in_buyers", "score_size_7d", "buyer_male_7d", "buyer_female_60d", "cvr_pv_main_1d", "refund_15d", "buyer_neutral_30d",
      "buyer_female_15d", "score_description_15d", "refund_7d", "buyer_neutral_15d", "click_uv_main_1d", "buyer_neutral_60d",
      "sales_productinfo_15d", "buyer_female_90d", "confirm_sales_15d", "score_60d", "sales_productinfo_7d", "score_quality_30d",
      "in_sales_15d", "confirm_sales_7d", "sales_main_30d", "confirm_sales_60d", "sales_productinfo_60d", "comment_cnt_60d",
      "male_rate_1d", "score_quality_15d", "confirm_sales_90d", "comment_cnt_90d", "confirm_sales_30d", "good_score_rate_90d",
      "in_sales_90d", "in_confirm_buyers", "score_30d", "in_sales_60d", "repurchase_rate_7d", "sales_main_60d", "sales_main",
      "cvr_uv_main_1d", "sales_main_15d", "good_score_rate_60d", "buyers_30d", "in_sales_30d", "sales_ctgr", "female_rate_1d",
      "score_15d", "sales_productinfo", "buyer_neutral_90d", "confirm_gmv_1d", "sales_ctgr_30d", "sales_ctgr_60d", "confirm_orders",
      "good_score_rate_30d", "sales_main_7d", "comment_cnt_1d", "confirm_buyer_male_60d", "confirm_price_1d", "orders_30d",
      "shipping_orders", "score_quality_7d", "refund_rate_1d", "buyers_90d", "buyers_60d", "confirm_buyer_neutral_30d",
      "add_uv_1d", "confirm_buyer_male_90d", "confirm_buyer_female_30d", "in_sales_7d", "sales_main_90d", "buyers_15d",
      "sales_productinfo_90d", "confirm_buyer_male_30d", "confirm_buyers", "confirm_buyer_neutral_60d", "confirm_buyer_female_60d",
      "confirm_neutral_rate_1d", "orders_15d", "orders_90d", "confirm_buyer_female_15d", "confirm_buyer_male_15d",
      "sales_search_1d", "confirm_buyer_female_90d", "sales_ctgr_15d", "sales_ctgr_90d", "shipping_orders_30d", "buyer_male_1d",
      "confirm_buyer_neutral_15d", "buyer_neutral_1d", "good_score_rate_15d", "orders_60d", "in_confirm_sales_90d",
      "confirm_buyer_neutral_90d", "sales_ctgr_7d", "confirm_buyer_female_7d", "buyers_7d", "buyer_female_1d",
      "shipping_orders_15d", "shipping_orders_90d", "in_confirm_sales_30d", "in_confirm_sales_15d", "in_confirm_sales_60d",
      "score_7d", "shipping_orders_60d", "confirm_male_rate_1d", "confirm_buyer_male_7d", "good_score_rate_7d",
      "in_buyers_90d", "score_size_1d", "confirm_buyer_neutral_7d", "confirm_price_unit_india_var_1d", "shipping_sales_1d",
      "orders_7d", "confirm_buyers_60d", "confirm_female_rate_1d", "score_description_1d", "in_buyers_30d", "confirm_buyers_30d",
      "confirm_buyers_90d", "in_buyers_60d", "confirm_orders_30d", "in_confirm_sales_7d", "sales_productinfo_1d",
      "confirm_orders_90d", "confirm_orders_60d", "refund_1d", "confirm_sales_1d", "shipping_orders_7d", "in_confirm_buyers_90d",
      "in_buyers_15d", "in_confirm_buyers_60d", "sales_main_1d", "confirm_orders_7d", "confirm_buyers_15d", "confirm_orders_15d",
      "buyers_1d", "confirm_buyers_7d", "in_confirm_buyers_30d", "in_buyers_7d", "score_quality_1d", "orders_1d",
      "sales_ctgr_1d", "confirm_buyer_neutral_1d", "in_confirm_buyers_15d", "confirm_buyer_female_1d", "in_sales_1d",
      "score_1d", "confirm_buyer_male_1d", "in_confirm_buyers_7d", "confirm_buyers_1d", "confirm_orders_1d",
      "good_score_rate_1d", "in_buyers_1d", "in_confirm_sales_1d", "shipping_orders_1d", "in_confirm_buyers_1d"
    )
    resDF = resDF.drop(remainColumns:_*)

    //    resDF = resDF.drop(removeColumnNames:_*)
    //    resDF.withColumn("ctr", udf_div(col("click")+3000, col("imp") + 110000))
    //      .withColumn("acr", udf_div(col("add")))


    //    val filteredDF = ds1.select(ds1.columns.filter(colName => !colsToRemove.contains(colName)).map(colname => new Column(colname)).toList: _*)
    def addIndex(df: DataFrame) = spark.createDataFrame(
      // Add index
      df.rdd.zipWithIndex.map{case (r, i) => Row.fromSeq(r.toSeq :+ i)},
      // Create schema
      StructType(df.schema.fields :+ StructField("_index", LongType, false))
    )
    resDF = addIndex(resDF.orderBy(rand))
    resDF = resDF.drop("_index")

    def func(column: Column) = column.cast(DoubleType)
    val itemColumnName = resDF.columns
    val itemFeaturesDF = resDF.select(itemColumnName.map(name => func(col(name))): _*)

    val featureColumnName = itemColumnName.drop(1).dropRight(1)
    println(featureColumnName)
    val vectorAssembler = new VectorAssembler().setInputCols(featureColumnName)
      .setOutputCol("features")

    val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)

    val xgbInput = vectorAssembler.transform(itemFeaturesDF).select($"label", $"pid", toDense($"features").alias("features"))

    val Array(trainingDF, testDF, evalDF) = xgbInput.randomSplit(Array(0.7, 0.2, 0.1), seed=123456789)
    trainingDF.repartition(256).persist(StorageLevel.MEMORY_AND_DISK)

    val xgbParam = Map(
      "eta" -> eta,
      "max_depth" -> nTreeDepth,
      "objective" -> "reg:squarederror",
      "subsample" -> 0.8,
      "num_round" -> numRound,
      "num_workers" -> numWorkers,
      "eval_metric" -> "rmse",
      "num_early_stopping_rounds" -> 20,
      "maximize_evaluation_metrics" -> false
    )

    //    println("xgbParam = " + xgbParam)

    val xgbRegression = new XGBoostRegressor(xgbParam).setFeaturesCol("features").setLabelCol("label").setEvalSets(Map("eval" -> evalDF))
    xgbRegression.setUseExternalMemory(true)
    //    val model = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("lable").setEvalSets(Map("eval" -> evalDF))
    //    model.setLeafPredictionCol("leafFeatures")
    //    xgbRegression.setPredictionCol("leafFeatures")


    val xgbRegressionModel = xgbRegression.fit(trainingDF)
    xgbRegressionModel.write.overwrite().save(path=xgbModelPath+"/model")
    //    XGBoostRegressor.load(path=xgbModelPath)

    //    xgbRegression.save(path = xgbModelPath)
    //    xgbRegressionModel.nativeBooster.saveModel(modelPath=xgbModelPath)
    val featureScoreMap = xgbRegressionModel.nativeBooster.getFeatureScore(featureColumnName)
    val featureScore = xgbRegressionModel.nativeBooster.getScore(featureColumnName, "gain")

    //    val featureScoreMap = xgbRegressionModel.nativeBooster.getScore(importanceType)
    val sortedScoreMap = featureScoreMap.toSeq.toDF("feature_name", "importance")
      .orderBy(col("importance").desc)
      .write
      .mode("overwrite")
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save(xgbModelPath + "/feature_importance")
    //    df.coalesce(1).write.option("header", "true").csv("sample_file.csv")

    //    println(sortedScoreMap)

    //    xgbRegression.booster.getFeatureScore()
    val train_results = xgbRegressionModel.transform(trainingDF)
    //    xgbRegressionModel.setLeafPredictionCol("leafFeatures")
    println(s"train results: \n ${train_results.select($"pid", $"label", $"prediction").show(10)}")
    val test_results = xgbRegressionModel.transform(testDF)
    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(test_results)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")
    //    test_results.withColumnRenamed()
    //    train_results.show()
    train_results.select($"pid", $"label", $"prediction").write.mode("overwrite").option("header", "true").csv(xgbModelPath + "/train_result")
    test_results.select($"pid", $"label", $"prediction").write.mode("overwrite").option("header", "true").csv(xgbModelPath + "/test_result")

  }
}