package cn.mesie
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
/**
 * 推荐
 */
object Recommend {
  def main(args: Array[String]){
    val sc = new SparkContext("local[2]", "Recommend")
    // 打开文件
    val rawData = sc.textFile("data/u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    //转换成Rating对象
    val ratings = rawRatings.map { case Array(user, movie, rating) =>Rating(user.toInt, movie.toInt, rating.toDouble) }
    //rank：对应ALS模型中的因子个数,iterations：对应运行时的迭代次数,lambda：该参数控制模型的正则化过程，从而控制模型的过拟合情况
    // rank、iterations和lambda参数的值分别为50、10和0.01
    //返回一个MatrixFactorizationModel对象。该对象将用户因子和物品因子分保存在一个(id,factor)对类型的RDD中。它们分别称作userFeatures和productFeatures
    val model = ALS.train(ratings, 50, 10, 0.01)
    //predict函数同样可以以(user, item)ID对类型的RDD对象为输入，这时它将为每一对都生成相应的预测得分
    val predictedRating = model.predict(789, 123)
    //要为某个用户生成前K个推荐物品， 可借助MatrixFactorizationModel 所提供的recommendProducts函数来实现。该函数需两个输入参数：user和num。其中user是用户ID，而num是要推荐的物品个数
    //返回值为预测得分最高的前num个物品。这些物品的序列按得分排序
    val topKRecs = model.recommendProducts(789, 10)
//    println(topKRecs.mkString("\n"))
    val movies = sc.textFile("data/u.item")
    val titles = movies.map(line => line.split("\\|").take(2)).map(array=> (array(0).toInt,array(1))).collectAsMap()
    println(titles(123))
    
  
  }
}