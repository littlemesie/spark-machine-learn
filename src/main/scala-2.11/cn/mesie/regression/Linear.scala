package cn.mesie.regression
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
/**
 * 线性回归
 */
object Linear {
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local[2]", "Linear")
    // 打开文件
    val rawData = sc.textFile("data/hour_noheader.csv")
    val num_data = rawData.count()
    val records = rawData.map(line => line.split(","))
    val first = records.first()
    //instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
    //处理数据
//     val data = records.map { r =>
//      val features = r(2).toDouble
//      LabeledPoint(1, Vectors.dense(features))
//    }
//    data.foreach( x => print(x + " "))   
//    println(num_data)
    records.cache()
  }
}