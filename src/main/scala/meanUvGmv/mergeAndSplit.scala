package meanUvGmv

import org.apache.spark.ml.attribute.{Attribute, NumericAttribute}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.StringType


class mergeAndSplit {

}
object mergeAndSplit{
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("mergeAndSplit")
      .master("local[2]")
      .getOrCreate()

    import spark.implicits._
    val df = Seq(("Ming", 20, 15552211521L), ("hong", 19, 13287994007L), ("zhi", 21, 15552211523L))
      .toDF("name", "age", "phone")
    println(df.show())

    val separator = ","
    println(df.map(_.toSeq.foldLeft("")(_ + separator + _).substring(1)).show())

    import org.apache.spark.sql.functions._
    println(df.select(concat_ws(separator, $"name", $"age", $"phone").cast(StringType).as("value")).show())

    def mergeCols(row: Row): String = {
      row.toSeq.foldLeft("")(_ + separator + _).substring(1)
    }

    val mergeColsUDF = udf(mergeCols _)
    println(df.select(mergeColsUDF(struct($"name", $"age", $"phone")).as("value")).show())

    var df_1 = Seq("Ming,20,15552211521", "hong,19,13287994007", "zhi,21,15552211523")
      .toDF("value")
    println(df_1.show())

    lazy val first = df_1.first()
    val numAttrs = first.toString().split(separator).length
    val attrs = Array.tabulate(numAttrs)(n=>"col_" + n)
    var newDF = df_1.withColumn("splitCols", split($"value", separator))
    attrs.zipWithIndex.foreach(x => {
      newDF = newDF.withColumn(x._1, $"splitCols".getItem(x._2))
    })
    println(newDF.show())

    val attributes: Array[Attribute] = {
      val numAttrs = first.toString().split(separator).length
      Array.tabulate(numAttrs)(i => NumericAttribute.defaultAttr.withName("value" + "_" + i))
    }

    val fieldCols = attributes.zipWithIndex.map(x => {
      val assembleFunc = udf{
        str: String =>
          str.split(separator)(x._2)
      }
      assembleFunc(df("value").cast(StringType)).as(x._1.name.get, x._1.toMetadata())
    })

    println(df.select(col("*") +: fieldCols: _*).show())

  }
}
