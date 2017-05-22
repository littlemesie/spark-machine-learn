package cn.mesie.cluster
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg._
import breeze.numerics.pow
import org.apache.spark.mllib.clustering.KMeans
/**
 * K-均值聚类
 */
object Kmeans {
  
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext("local", "Kmeans")
    val movies = sc.textFile("data/u.item")
    
    val genres = sc.textFile("data/u.genre")
    //得到具体的<题材，索引>键值对的map
    //filter(!_.isEmpty) 处理空行，防止出现异常
    val genreMap = genres.filter(!_.isEmpty).map(line => line.split("\\|")).map(array => (array(1), array(0))).collectAsMap
    println(genreMap)
    //take(i)获取i行数据
//    genres.take(5).foreach(println)
    //提取 （电影Id, 标题，题材）
    val titlesAndGenres = movies.map(_.split("\\|")).map { array =>
      
      val genres = array.toSeq.slice(5, array.size)
      val genresAssigned = genres.zipWithIndex.filter { case (g, idx) =>
        g == "1"
      }.map { case (g, idx) =>
        genreMap(idx.toString)//得到题材
      }
      (array(0).toInt, (array(1), genresAssigned))
    }
    println(titlesAndGenres.first)
    println(movies.first)
    //训练一个新的推荐模型
    val rawData = sc.textFile("data/u.data")
    val rawRatings = rawData.map(_.split("\t").take(3))
    val ratings = rawRatings.map{ case Array(user, movie, rating) =>Rating(user.toInt, movie.toInt, rating.toDouble) }
    ratings.cache
    val alsModel = ALS.train(ratings, 50, 10, 0.1) //返回两pairRDD：Features/productFeatrures
    //提取相关因素转化到Vector中作为聚类模型训练输入
    val movieFactors = alsModel.productFeatures.map { case (id, factor) => (id, Vectors.dense(factor)) }
    val movieVectors = movieFactors.map(_._2)
    val userFactors = alsModel.userFeatures.map { case (id, factor) =>(id, Vectors.dense(factor)) }
    val userVectors = userFactors.map(_._2)
    //归一化
    val movieMatrix = new RowMatrix(movieVectors)
    val movieMatrixSummary = movieMatrix.computeColumnSummaryStatistics()
    val userMatrix = new RowMatrix(userVectors)
    val userMatrixSummary = userMatrix.computeColumnSummaryStatistics()
    println("Movie factors mean: " + movieMatrixSummary.mean)
    println("Movie factors variance: " + movieMatrixSummary.variance)
    println("User factors mean: " + userMatrixSummary.mean)
    println("User factors variance: " + userMatrixSummary.variance)
    //训练模型
    val numClusters = 5 
    val numIterations = 100 //最大迭代次数
    val numRuns = 3 //算法并发运行数目，从多个起点并发执行，最后选择最佳结果，多次训练可有效找倒最优模型
    val movieClusterModel = KMeans.train(movieVectors, numClusters,numIterations, numRuns)
//    val userClusterModel = KMeans.train(userVectors, numClusters,numIterations, numRuns)
    println(movieClusterModel)
    //使用聚类模型进行预测
    //单样本预测
    val movie1 = movieVectors.first
    val movieCluster = movieClusterModel.predict(movie1)
    println(movieCluster)
    val predictions = movieClusterModel.predict(movieVectors) // 多样本预测
    println(predictions.take(10).mkString(","))
    /**7.使用数据集解释类别预测 */

    def computeDistance(v1: DenseVector[Double], v2: DenseVector[Double]) = pow(v1 - v2, 2).sum
    
    //对-每个电影-计算其-特征向量-与-所属类簇中心向量的-距离-
    val titlesWithFactors = titlesAndGenres.join(movieFactors) //不是movieVectors(因为titlesAndGenres是kv对)
    
    val moviesAssigned = titlesWithFactors.map { case (id, ((title, genres),vector)) =>
        val pred = movieClusterModel.predict(vector)
        val clusterCentre = movieClusterModel.clusterCenters(pred)
        val dist = computeDistance(DenseVector(clusterCentre.toArray), DenseVector(vector.toArray))//中心->特征向量
        (id, title, genres.mkString(" "), pred, dist)
    }
    val clusterAssignments = moviesAssigned.groupBy {case (id, title, genres, cluster, dist) =>
      cluster
    }.collectAsMap() //键->类簇标识，值->电影和相关信息的组合
    
    //接着，枚举每个类簇并输出距离类中心最近的前20部电影
    for ( (k,v) <- clusterAssignments.toSeq.sortBy(_._1)) {
      println(s"Cluster $k: ")
      val m = v.toSeq.sortBy(_._5)
      println(m.take(20).map { case (_, title, genres, _, d) =>
        (title, genres, d)
      }.mkString("\n")
      )
      println("====/n")
    }
    
    
  }
}