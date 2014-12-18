/**
 * Created by ian on 12/17/14.
 *
 * @author Ian Tabolt <iantabolt@gmail.com>
 */
case class FeatureNormalizer(means: Vector[Double], ranges: Vector[Double])
  extends (Vector[Double]=>Vector[Double]) {
  def apply(targetFeatures: Vector[Double]) = {
    val zeroMeans = (targetFeatures,means).zipped.map(_ - _)
    (zeroMeans,ranges).zipped.map(_ / _)
  }
  override def toString =
    "FeatureNormalizer(means=" + means + ", ranges=" + ranges + ")"
}

object FeatureNormalizer {
  def fromTrainData(trainDataFeatures: List[Vector[Double]]): FeatureNormalizer = {
    val m = trainDataFeatures.size
    val n = trainDataFeatures.head.size
    val means = trainDataFeatures.reduce((f1,f2)=>(f1,f2).zipped.map(_+_)).map(_ / m)
    val ranges: Vector[Double] = (0 until n).map { i =>
      val slice = trainDataFeatures.map(_(i))
      slice.max - slice.min
    }.toVector
    FeatureNormalizer(means,ranges)
  }
}
