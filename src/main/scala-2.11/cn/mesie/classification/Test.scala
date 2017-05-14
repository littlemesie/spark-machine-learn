package cn.mesie.classification
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
/**
 * 
 */
object Test {
  def main(args: Array[String]){
    val sc = new SparkContext("local[2]", "Test")
    // 打开文件
    val rawData = sc.textFile("data/train.tsv")
    val records = rawData.map(line => line.split("\t"))
//    records.first().foreach(println(_))
//    records.first()
  }

}