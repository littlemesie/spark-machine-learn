package cn.mesie.classification
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.feature.StandardScaler
/**
 * 训练模型
 */
object TrainModel {
  def main(args: Array[String]): Unit = {
     val sc = new SparkContext("local[2]", "Test")
    // 打开文件
    val rawData = sc.textFile("data/train.tsv")
    val records = rawData.map(line => line.split("\t"))
    //在清理和处理缺失数据后，我们提取最后一列的标记变量以及第5列到第25列的特征矩阵。
    val data = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d ==
      "?") 0.0 else d.toDouble)
      LabeledPoint(label, Vectors.dense(features))
    }
    //将负值设置为0,主要用于贝叶斯算法
    val nbData = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val features = trimmed.slice(4, r.size - 1).map(d => if (d ==
      "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
      LabeledPoint(label, Vectors.dense(features))
    }
    //数据的缓存
    data.cache
    val numData = data.count 
    val numIterations = 10
    val maxTreeDepth = 5
    val lrModel = LogisticRegressionWithSGD.train(data, numIterations)
//    val svmModel = SVMWithSGD.train(data, numIterations)
//    val nbModel = NaiveBayes.train(nbData)
//    val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)
    //预测
//    val dataPoint = data.first
//    val prediction = lrModel.predict(dataPoint.features)
    val predictions = lrModel.predict(data.map(lp => lp.features))
    val lrTotalCorrect = data.map { point =>
      if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    //正确率
    val lrAccuracy = lrTotalCorrect / data.count
    //计算二分类的PR和ROC曲线下的面积
//    val metrics = Seq(lrModel, svmModel).map { model =>
//      val scoreAndLabels = data.map { point =>
//        (model.predict(point.features), point.label)
//      }
//      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
//      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
//    }
//    
//    val nbMetrics = Seq(nbModel).map{ model =>
//      val scoreAndLabels = nbData.map { point =>
//      val score = model.predict(point.features)
//      (if (score > 0.5) 1.0 else 0.0, point.label)
//      }
//      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
//      (model.getClass.getSimpleName, metrics.areaUnderPR,
//      metrics.areaUnderROC)
//    }
//    
//    val dtMetrics = Seq(dtModel).map{ model =>
//      val scoreAndLabels = data.map { point =>
//        val score = model.predict(point.features)
//        (if (score > 0.5) 1.0 else 0.0, point.label)
//      }
//      val metrics = new BinaryClassificationMetrics(scoreAndLabels)
//      (model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
//    }
//    val allMetrics = metrics ++ nbMetrics ++ dtMetrics
//    allMetrics.foreach{ case (m, pr, roc) =>
//      println(f"$m, Area under PR: ${pr*100.0}%2.4f%%, Area under
//      ROC: ${roc * 100.0}%2.4f%%")
//    }
    //改进模型性能以及参数调优
    val vectors = data.map(lp => lp.features)
    //特征向量用RowMatrix类表示成MLlib中的分布矩阵
    val matrix = new RowMatrix(vectors)
    val matrixSummary = matrix.computeColumnSummaryStatistics()
    //将输入向量传到转换函数，并且返回归一化的向量
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
    val scaledData = data.map(lp => LabeledPoint(lp.label,
      scaler.transform(lp.features)))
    //使用标准化的数据重新训练模型
    val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
    val lrTotalCorrectScaled = scaledData.map { point =>
      if (lrModelScaled.predict(point.features) == point.label) 1 else 0
    }.sum
    //正确率
    val lrAccuracyScaled = lrTotalCorrectScaled / numData
    val lrPredictionsVsTrue = scaledData.map { point =>
      (lrModelScaled.predict(point.features), point.label)
    }
    val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
    val lrPr = lrMetricsScaled.areaUnderPR
    val lrRoc = lrMetricsScaled.areaUnderROC
//    println(lrAccuracyScaled)
    
    //其它特征
    val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val numCategories = categories.size
//    println(categories)
    //创建一个14的向量来表示类别特征
    val dataCategories = records.map { r =>
      val trimmed = r.map(_.replaceAll("\"", ""))
      val label = trimmed(r.size - 1).toInt
      val categoryIdx = categories(r(3))
      val categoryFeatures = Array.ofDim[Double](numCategories)
      categoryFeatures(categoryIdx) = 1.0
      val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if
      (d == "?") 0.0 else d.toDouble)
      val features = categoryFeatures ++ otherFeatures
      LabeledPoint(label, Vectors.dense(features))
    }
//    println(dataCategories.first)
    //数据归一化处理
    val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
    val scaledDataCats = dataCategories.map(lp =>LabeledPoint(lp.label, scalerCats.transform(lp.features)))
    //训练模型
    val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats,numIterations)
    val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
      if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
    val lrPredictionsVsTrueCats = scaledDataCats.map { point =>
      (lrModelScaledCats.predict(point.features), point.label)
    }
    val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
    val lrPrCats = lrMetricsScaledCats.areaUnderPR
    val lrRocCats = lrMetricsScaledCats.areaUnderROC
    println(lrAccuracyScaledCats)
//   
  }
}