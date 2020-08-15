package meanUvGmv

import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{col, rand, udf}
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}
import org.apache.spark.storage.StorageLevel

class xgbTrainV4{

}


object xgbTrainV4{
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
    val spark = SparkSession.builder().appName("search_jixiaozhan_gmvTrainV4").enableHiveSupport().getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("WARN")
    var log_date = log_dates(0)
    var label_log_date = label_log_dates(0)
    var resFeaturesDF = spark.sql(f"select * from search_algorithm.product_score_features where date_id='${log_date}'")
    resFeaturesDF = resFeaturesDF.where("is_sale == 1")
    resFeaturesDF = resFeaturesDF.drop("date_id")
    resFeaturesDF = resFeaturesDF.dropDuplicates("pid")

    var resLabelDF = spark.sql(f"select * from search_algorithm.offlineTotalAvgGmv where date_id='${label_log_date}'")
    resLabelDF = resLabelDF.drop("date_id")
    resLabelDF = resLabelDF.where("total_uv > 0")
    resLabelDF = resLabelDF.drop("total_uv")
    val resLabelDF_1  = resLabelDF.where("mean_uv_gmv == 0").sample(sample_rate)
    val resLabelDF_2 = resLabelDF.where("mean_uv_gmv > 0")
    resLabelDF = resLabelDF_1.union(resLabelDF_2)
    resLabelDF = resLabelDF.drop("mean_uv_gmv")
    var resDF = resFeaturesDF.join(resLabelDF, Seq("pid"), "inner")

    for (i <- 1 to (log_dates.length - 1)){
      log_date = log_dates(i)
      var featuresDF = spark.sql(f"select * from search_algorithm.product_score_features where date_id='${log_date}'")
      featuresDF = featuresDF.where("is_sale == 1")
      featuresDF = featuresDF.drop("date_id")
      featuresDF = featuresDF.dropDuplicates("pid")

      label_log_date = label_log_dates(i)
      var resLabelDF = spark.sql(f"select * from search_algorithm.offlineTotalAvgGmv where date_id='${label_log_date}'")
      resLabelDF = resLabelDF.drop("date_id")
      resLabelDF = resLabelDF.where("total_uv > 0")
      resLabelDF = resLabelDF.drop("total_uv")
      val resLabelDF_1  = resLabelDF.where("mean_uv_gmv == 0").sample(sample_rate)
      val resLabelDF_2 = resLabelDF.where("mean_uv_gmv > 0")
      resLabelDF = resLabelDF_1.union(resLabelDF_2)
      resLabelDF = resLabelDF.drop("mean_uv_gmv")
      val resDF1 = featuresDF.join(resLabelDF, Seq("pid"), "inner")
      resDF = resDF.union(resDF1)
    }

   def addIndex(df: DataFrame) = spark.createDataFrame(
      // Add index
      df.rdd.zipWithIndex.map{case (r, i) => Row.fromSeq(r.toSeq :+ i)},
      // Create schema
      StructType(df.schema.fields :+ StructField("_index", LongType, false))
    )
    resDF = resDF
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
    val xgbRegression = new XGBoostRegressor(xgbParam).setFeaturesCol("features").setLabelCol("label").setEvalSets(Map("eval" -> evalDF))
    xgbRegression.setUseExternalMemory(true)

    val xgbRegressionModel = xgbRegression.fit(trainingDF)
    xgbRegressionModel.write.overwrite().save(path=xgbModelPath+"/model")
    val featureScoreMap = xgbRegressionModel.nativeBooster.getFeatureScore(featureColumnName)
    val featureScore = xgbRegressionModel.nativeBooster.getScore(featureColumnName, "gain")

    val sortedScoreMap = featureScoreMap.toSeq.toDF("feature_name", "importance")
      .orderBy(col("importance").desc)
      .write
      .mode("overwrite")
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save(xgbModelPath + "/feature_importance")

    val train_results = xgbRegressionModel.transform(trainingDF)
    println(s"train results: \n ${train_results.select($"pid", $"label", $"prediction").show(10)}")
    val test_results = xgbRegressionModel.transform(testDF)
    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(test_results)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    train_results.select($"pid", $"label", $"prediction").write.mode("overwrite").option("header", "true").csv(xgbModelPath + "/train_result")
    test_results.select($"pid", $"label", $"prediction").write.mode("overwrite").option("header", "true").csv(xgbModelPath + "/test_result")

  }
}