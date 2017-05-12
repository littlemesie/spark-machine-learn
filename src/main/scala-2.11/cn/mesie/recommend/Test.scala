package cn.mesie.recommend
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
object Test {
  def main(args: Array[String]){
    val sc = new SparkContext("local[2]", "Test")
    // 打开文件
    val rawData = sc.textFile("data/u.data")
    val movies = sc.textFile("data/u.item")
    val rawRatings = rawData.map(_.split("\t").take(3))
    //转换成Rating对象
    val ratings = rawRatings.map { case Array(user, movie, rating) =>Rating(user.toInt, movie.toInt, rating.toDouble) }
   
    //找出用户789评价的电影
    val moviesForUser = ratings.keyBy(_.user).lookup(789)
    println(moviesForUser.size)
    
    val titles = movies.map(line => line.split("\\|").take(2)).map(array=> (array(0).toInt,array(1))).collectAsMap()
    
    // 评级最高的前10部电影
    moviesForUser.sortBy(-_.rating).take(10).map(rating => (titles(rating.product),
    rating.rating)).foreach(println)
    
    println(titles(123))
  }
}