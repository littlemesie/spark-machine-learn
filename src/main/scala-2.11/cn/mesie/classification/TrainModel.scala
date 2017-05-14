package cn.mesie.classification
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
/**
 * 训练模型
 */
object TrainModel {
  def main(args: Array[String]): Unit = {
     val sc = new SparkContext("local[2]", "Test")
    // 打开文件
    val rawData = sc.textFile("data/train.tsv")
    val records = rawData.map(line => line.split("\t"))
    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d ==
      "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    //数据的缓存
    data.cache
    val numData = data.count 
    println(numData)
  }
}