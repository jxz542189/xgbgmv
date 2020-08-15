package sales7d

import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel, XGBoostRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.{rand, _}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.storage.StorageLevel


class Sales7dXgbTrain {

}

object Sales7dXgbTrain {
  def main(args: Array[String]): Unit = {

    if (args.length < 10) {
      println("input args: samplesDir, learning_rate, num_round, max_depth," +
        " num_early_stopping_rounds, num_workers, eval_metric, convertLabel, modelOutputDir, logLevel")
      sys.exit(1)
    }

    // 解析参数
    val samplesDir = args(0)
    val learning_rate = args(1).toFloat
    val num_round = args(2).toInt
    val max_depth = args(3).toInt
    val num_early_stopping_rounds = args(4).toInt
    val num_workers = args(5).toInt
    val eval_metric = args(6)
    val convertLabel = args(7).toBoolean
    val modelOutputDir = args(8)
    val logLevel = args(9)

    val spark = SparkSession.builder().appName("search_dupeng_sales_7d_train").enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel(logLevel)
    import spark.implicits._

    /**
      * 读取所有样本
      * @param baseDir: 样本目录
      * @return (训练样本, 测试样本, 验证样本), DataFrame: [pid, date_id, label, label_converted, features]
      */
    def getAllSamples(baseDir: String): (DataFrame, DataFrame, DataFrame, Array[String]) = {
      // 读取features数据（过滤掉不在架的商品）, [pid, feature1, feature2, ..., date_id]
      println("read features...")
      val featuresDF = spark.sql(f"select * from search_algorithm.gmv_item_profile_feature where is_sale = 1")
      val featureColNames = featuresDF.columns.drop(1).dropRight(1)

      // 读取3个样本集, [pid, date_id, label, label_converted, features]
      val trainingSamples = getSamples(baseDir, "training", featuresDF)
      val testSamples = getSamples(baseDir, "test", featuresDF)
      val validationSamples = getSamples(baseDir, "validation", featuresDF)

      // 打印总样本数
      val totalCount = trainingSamples.count() + testSamples.count() + validationSamples.count()
      val aboveZeroCount = trainingSamples.where("label > 0").count() +
                           testSamples.where("label > 0").count() +
                           validationSamples.where("label > 0").count()
      val zeroCount = trainingSamples.where("label == 0").count() +
                      testSamples.where("label == 0").count() +
                      validationSamples.where("label == 0").count()
      println(s"total samples count: $totalCount, label>0: $aboveZeroCount, label=0: $zeroCount")

      (trainingSamples, testSamples, validationSamples, featureColNames)
    }

    /**
      * 读取样本并填充features
      * @param baseDir: 样本目录
      * @param sampleType: 样本类别
      * @param featuresDF: 特征数据
      * @return 样本DataFrame: [pid, date_id, label, label_converted, features]
      */
    def getSamples(baseDir: String, sampleType: String, featuresDF: DataFrame) : DataFrame = {
      println(s"read $sampleType samples...")

      // 读样本数据，[pid, date_id, label, label_converted]
      var df = spark
        .read
        .format("csv")
        .option("inferSchema", "true")
        .option("header", "true")
        .load(s"$baseDir/${sampleType}_samples.csv")
        .withColumn("date_id_str", date_format(col("date_id"), "yyyy-MM-dd"))
        .drop("date_id")
        .withColumnRenamed("date_id_str", "date_id")
      val columns = Array("pid", "date_id") ++ df.columns.slice(1, df.columns.length - 1)
      df = df.select(columns.map(c => col(c)): _*)

      // join, [pid, date_id, label, label_converted, feature1, ...]
      var samplesDF = df.join(featuresDF, Seq("pid", "date_id"), "inner")

      // 处理features, [pid, date_id, label, label_converted, features]
      samplesDF = processFeatures(samplesDF)

      // 打印样本数
      println(s"$sampleType samples count: ${samplesDF.count()}," +
              s" sales > 0: ${samplesDF.where("label > 0").count()}," +
              s" sales = 0: ${samplesDF.where("label == 0").count()}")

      samplesDF
    }

    /**
      * 处理样本features
      * @param samplesDF: 处理前的样本DF，[pid, date_id, label, label_converted, feature1, ...]
      * @return 处理后的样本DF，[pid, date_id, label, label_converted, features]
      */
    def processFeatures(samplesDF: DataFrame) : DataFrame = {
      // 1. 转Double类型
      val tempColNames = Array("pid", "date_id")
      def func(colName: String) = {
        if (tempColNames.contains(colName)) {
          col(colName)
        } else {
          col(colName).cast(DoubleType)
        }
      }
      val itemColNames = samplesDF.columns
      var itemDF = samplesDF.select(itemColNames.map(name => func(name)): _*)

      // 2. VectorAssembler, [pid, date_id, label, label_converted, features]
      // drop: [pid, date_id, label, label_converted] or [pid, date_id, label]
      val featureColNames = if (convertLabel) itemColNames.drop(4) else itemColNames.drop(3)
      println(s"features count: ${featureColNames.length}")
      val vectorAssembler = new VectorAssembler().setInputCols(featureColNames).setOutputCol("features")
      itemDF = vectorAssembler.transform(itemDF)

      // 3. toDense
      val toDense = udf((v: org.apache.spark.ml.linalg.Vector) => v.toDense)
      var xgbInput = spark.emptyDataFrame
      if (convertLabel) {
        // [pid, date_id, label, label_converted, features]
        xgbInput = itemDF.select($"pid", $"date_id", $"label", $"label_converted", toDense($"features").alias("features"))
      } else {
        // [pid, date_id, label, features]
        xgbInput = itemDF.select($"pid", $"date_id", $"label", toDense($"features").alias("features"))
      }
      println(s"xgbInput: \n ${xgbInput.where("label > 10").orderBy(rand).show(10)}")

      xgbInput
    }

    /**
      * 训练模型
      * @param trainingSamples: 训练样本, [pid, date_id, label, label_converted, features]
      * @param testSamples: 测试样本, [pid, date_id, label, label_converted, features]
      * @param validationSamples: 验证样本, [pid, date_id, label, label_converted, features]
      * @param modelOutputDir: 模型输出目录
      * @return xgb回归模型
      */
    def train(trainingSamples: DataFrame, testSamples: DataFrame, validationSamples: DataFrame, featureColNames: Array[String], modelOutputDir: String) : XGBoostRegressionModel = {
      trainingSamples.persist(StorageLevel.MEMORY_AND_DISK)

      val labelColName = if (convertLabel) "label_converted" else "label"

      // 实例化xgb回归器
      val xgbParam = Map(
        "eta" -> learning_rate,
        "max_depth" -> max_depth,
        "objective" -> "reg:squarederror",
        "subsample" -> 0.8,
        "num_round" -> num_round,
        "num_workers" -> num_workers,
        "eval_metric" -> eval_metric,
        "num_early_stopping_rounds" -> num_early_stopping_rounds,
        "maximize_evaluation_metrics" -> false
      )
      val xgbRegressor = new XGBoostRegressor(xgbParam).setFeaturesCol("features").setLabelCol(labelColName).setEvalSets(Map("eval" -> validationSamples))
      xgbRegressor.setUseExternalMemory(true)

      // 训练并保存xgb回归模型
      println("train...")
      val xgbRegressionModel = xgbRegressor.fit(trainingSamples)
      xgbRegressionModel.write.overwrite().save(path=modelOutputDir+"/model")

      // feature importance
      val featureScore = xgbRegressionModel
        .nativeBooster
        .getFeatureScore(featureColNames)
        .toSeq
        .toDF("feature_name", "importance")
        .orderBy(col("importance").desc)
      println(s"feature importance: \n ${featureScore.show(30)}")

      featureScore
        .write
        .mode("overwrite")
        .format("com.databricks.spark.csv")
        .option("header", "true")
        .save(modelOutputDir + "/feature_importance")

      xgbRegressionModel
    }

    /**
      * 在所有样本集上评估模型
      * @param xgbModel: xgb回归模型
      * @param trainingSamples: 训练样本
      * @param testSamples: 测试样本
      * @param validationSamples: 验证样本
      * @param modelOutputDir: 模型输出目录
      */
    def evaluateAll(xgbModel: XGBoostRegressionModel,
                    trainingSamples: DataFrame,
                    testSamples: DataFrame,
                    validationSamples: DataFrame,
                    modelOutputDir: String): Unit = {
      evaluate(xgbModel, trainingSamples, "training", modelOutputDir)
      evaluate(xgbModel, testSamples, "test", modelOutputDir)
      evaluate(xgbModel, validationSamples, "validation", modelOutputDir)
    }

    /**
      * 在一个样本集上评估模型
      * @param xgbModel: xgb回归模型
      * @param samples: 样本
      * @param sampleType: 样本类别
      * @param modelOutputDir: 模型输出目录
      */
    def evaluate(xgbModel: XGBoostRegressionModel, samples: DataFrame, sampleType: String, modelOutputDir: String): Unit = {
      // 预测，[pid, date_id, label, label_converted, features, prediction]
      println(s"evalute $sampleType set...")
      var results = xgbModel.transform(samples)

      // 将预测结果值还原，[pid, date_id, label, label_converted, features, prediction, prediction_recovered]
      if (convertLabel) {
        results = results.withColumn("prediction_recovered", expm1("prediction"))
      }
      println(s"$sampleType results: \n ${results.where("label > 10").orderBy(rand).show(100)}")

      // 保存预测结果，注意：这里不能用coalesce(resPartitions)，否则会导致数据丢失和错乱！
      // [pid, date_id, label, label_converted, prediction, prediction_recovered] or [pid, date_id, label, prediction]
      println(s"save $sampleType results")
      results
        .drop("features")
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(modelOutputDir + s"/${sampleType}_result")

      // 计算rmse指标
      val predictioColName = if (convertLabel) "prediction_recovered" else "prediction"
      val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol(predictioColName).setMetricName("rmse")
      val rmse = evaluator.evaluate(results)
      println(s"Root Mean Squared Error (RMSE) on $sampleType set: $rmse")
    }

    // 读取样本
    val (trainingSamples, testSamples, validationSamples, featureColNames) = getAllSamples(samplesDir)

    // 训练xgb回归模型
    val xgbRegressionModel = train(trainingSamples, testSamples, validationSamples, featureColNames, modelOutputDir)

    // 评估模型
    evaluateAll(xgbRegressionModel, trainingSamples, testSamples, validationSamples, modelOutputDir)
  }
}