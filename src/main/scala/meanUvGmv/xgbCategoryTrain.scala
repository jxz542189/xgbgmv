package meanUvGmv

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostRegressor}
import org.apache.spark.api.java.function.MapFunction
import org.apache.spark.{SparkConf, SparkContext}
//import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{SparkSession, _}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.rand


class xgbCategoryTrain {

}

object xgbCategoryTrain{
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
    val spark = SparkSession.builder().appName("search_jixiaozhan_gmvTrain").enableHiveSupport().getOrCreate()
    //    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    var log_date = log_dates(0)
    var label_log_date = label_log_dates(0)
    var resFeaturesDF = spark.sql(f"select * from search_algorithm.gmv_item_profile_feature where date_id='${log_date}'")
    resFeaturesDF = resFeaturesDF.where("is_sale == 1")
    resFeaturesDF = resFeaturesDF.drop("date_id")
    var resLabelDF = spark.sql(f"select * from search_algorithm.offlineAvgUvGmvCategory where date_id='${label_log_date}'")
    resLabelDF = resLabelDF.drop("date_id")
    resLabelDF = resLabelDF.where("ctgr_total_uv > 0")
    resLabelDF = resLabelDF.drop("ctgr_total_uv")
    val resLabelDF_1  = resLabelDF.where("ctgr_mean_uv_gmv == 0").sample(sample_rate)
    val resLabelDF_2 = resLabelDF.where("ctgr_mean_uv_gmv > 0")
    resLabelDF = resLabelDF_1.union(resLabelDF_2)
    resLabelDF = resLabelDF.drop("ctgr_mean_uv_gmv")
    resLabelDF = resLabelDF.select("pid", "ctgr_label")
    var resDF = resFeaturesDF.join(resLabelDF, Seq("pid"), "inner")
    //    println(resDF.show(1))

    for (i <- 1 to (log_dates.length - 1)){
      log_date = log_dates(i)
      var featuresDF = spark.sql(f"select * from search_algorithm.gmv_item_profile_feature where date_id='${log_date}'")
      featuresDF = featuresDF.where("is_sale == 1")
      featuresDF = featuresDF.drop("date_id")
      label_log_date = label_log_dates(i)
      var resLabelDF = spark.sql(f"select * from search_algorithm.offlineAvgUvGmvCategory where date_id='${label_log_date}'")
      resLabelDF = resLabelDF.drop("date_id")
      resLabelDF = resLabelDF.where("ctgr_total_uv > 0")
      resLabelDF = resLabelDF.drop("ctgr_total_uv")
      val resLabelDF_1  = resLabelDF.where("ctgr_mean_uv_gmv == 0").sample(sample_rate)
      val resLabelDF_2 = resLabelDF.where("ctgr_mean_uv_gmv > 0")
      resLabelDF = resLabelDF_1.union(resLabelDF_2)
      resLabelDF = resLabelDF.drop("ctgr_mean_uv_gmv")
      resLabelDF = resLabelDF.select("pid", "ctgr_label")
      val resDF1 = featuresDF.join(resLabelDF, Seq("pid"), "inner")
      //      println(resDF1.)
      resDF = resDF.union(resDF1)
    }

    resDF = resDF.withColumnRenamed("ctgr_label", "label")

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