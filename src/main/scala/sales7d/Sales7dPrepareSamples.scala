package sales7d

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.joda.time.DateTime
import org.joda.time.format.DateTimeFormat


class Sales7dPrepareSamples {

}

object Sales7dPrepareSamples {
  def main(args: Array[String]): Unit = {

    if (args.length < 6) {
      println("input args: log_dates, sample_rate," +
        " useMean, convertLabel, saveDir, logLevel")
      sys.exit(1)
    }

    // 解析参数
    val log_dates = args(0).split(',')
    val sample_rate = args(1).toFloat
    val useMean = args(2).toBoolean
    val convertLabel = args(3).toBoolean
    val saveDir = args(4)
    val logLevel = args(5)

    val spark = SparkSession.builder().appName("search_dupeng_sales7d_samples").enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel(logLevel)

    // 获取feature数据及label数据，并将所有数据合并
    var resDF = spark.emptyDataFrame
    for (i <- log_dates.indices) {
      val log_date = log_dates(i)

      // 读取feature数据（过滤掉不在架的商品）, [pid, date_id]
      val featuresDF = spark.sql(f"select pid, date_id from search_algorithm.gmv_item_profile_feature where is_sale = 1 and date_id='$log_date'")
      //      println(featuresDF.show(1))

      // 读取label数据
      // 数据过滤：label对应的7天需要都在架（简单起见，目前先只用第7天在架的过滤条件）
      val end_date = DateTime.parse(log_date, DateTimeFormat.forPattern("yyyy-MM-dd")).plusDays(7).toString("yyyy-MM-dd")
      var labelDF = spark.sql(
        f"select t1.* from " +
          f"(select * from search_algorithm.sales_7d_label where date_id = '$log_date') t1 " +
          f"join (select pid from search_algorithm.gmv_item_profile_feature where is_sale = 1 and date_id = '$end_date') t2 " +
          f"on t1.pid = t2.pid"
      )
      labelDF = labelDF.drop("date_id").withColumnRenamed("sales_7d", "label")  // [pid, label]

      // label转换为7天平均销量
      if (useMean) {
        def getMean(colName: String) = {
          col(colName) / 7.0
        }

        labelDF = labelDF
          .withColumn("label_mean", getMean("label"))
          .drop("label")
          .withColumnRenamed("label_mean", "label")
      }

      // feature和label join, [pid, date_id, label]
      var curDF = featuresDF.join(labelDF, Seq("pid"), "inner")

      // 根据label对样本采样
      val curDF_1  = curDF.where("label == 0").sample(sample_rate)
      val curDF_2 = curDF.where("label > 0")
      curDF = curDF_1.union(curDF_2)

      // 合并不同日期的数据
      if (i == 0)
        resDF = curDF
      else
        resDF = resDF.union(curDF)

      // 打印每天样本数
      println(s"$log_date samples count: ${curDF.count()}," +
        s" sales > 0: ${curDF.where("label > 0").count()}," +
        s" sales = 0: ${curDF.where("label == 0").count()}")
    }
    // 打印总样本数
    println(s"total samples count: ${resDF.count()}," +
      s" sales > 0: ${resDF.where("label > 0").count()}," +
      s" sales = 0: ${resDF.where("label == 0").count()}")

    // 根据是否转换标签，用不同标签列名
    val labelName = "label"
    var labelNameUsed = labelName
    if (convertLabel) {
      labelNameUsed = "label_converted"
    }

    // 将label转换为log(x+1), [pid, date_id, label, label_converted]
    if (convertLabel) {
      resDF = resDF.withColumn(labelNameUsed, log1p(labelName))
    }

    // 数据集split
    val Array(trainingDF, testDF, validationDF) = resDF.randomSplit(Array(0.7, 0.2, 0.1), seed=123456789)
    println(s"training samples: ${trainingDF.count()}, " +
            s"test samples: ${testDF.count()}, " +
            s"validation samples: ${validationDF.count()}")

    // 保存训练集、测试集、验证集
    trainingDF.write.mode("overwrite").option("header", "true").csv(saveDir + "/training_set")
    testDF.write.mode("overwrite").option("header", "true").csv(saveDir + "/test_set")
    validationDF.write.mode("overwrite").option("header", "true").csv(saveDir + "/validation_set")
  }
}