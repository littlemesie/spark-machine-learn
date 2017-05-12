package cn.mesie.recommend
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.jblas.DoubleMatrix
/**
 * 基于物品的推荐
 */
object ItemRecom {
  /**
   * 余弦相似度计算函数
   */
  def cosineSimilarity(vec1: DoubleMatrix, vec2: DoubleMatrix): Double = {
    vec1.dot(vec2) / (vec1.norm2() * vec2.norm2())
  }
  
  def main(args: Array[String])= {
    val sc = new SparkContext("local[2]", "Recommend")
    // 打开文件
    val rawData = sc.textFile("data/u.data")
    val movies = sc.textFile("data/u.item")
    val rawRatings = rawData.map(_.split("\t").take(3))
    
    val titles = movies.map(line => line.split("\\|").take(2)).map(array=> (array(0).toInt,array(1))).collectAsMap()
    //转换成Rating对象
    val ratings = rawRatings.map { case Array(user, movie, rating) =>Rating(user.toInt, movie.toInt, rating.toDouble) }
    val model = ALS.train(ratings, 50, 10, 0.01)
    //以用户567为例来计算余弦显示度
    val itemId = 567
    val itemFactor = model.productFeatures.lookup(itemId).head
    val itemVector = new DoubleMatrix(itemFactor)
    val s = cosineSimilarity(itemVector, itemVector)
    println(s)
    //计算各个物品的相似度
    val sims = model.productFeatures.map{ case (id, factor) => 
    	val factorVector = new DoubleMatrix(factor)
    	val sim = cosineSimilarity(factorVector, itemVector)
    	(id, sim)
    }
    
    val K = 10
    val sortedSims = sims.top(K)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
//    println(sortedSims.mkString("\n"))
//    
//    println(titles(itemId))
    val sortedSims2 = sims.top(K + 1)(Ordering.by[(Int, Double), Double] { case (id, similarity) => similarity })
    val ss = sortedSims2.slice(1, 11).map{ case (id, sim) => (titles(id), sim) }
    println(ss.mkString("\n"))
    
  }
  
}