import java.net.URL
import scala.util.Random

case class LinearModel(params: Vector[Double]) {
  lazy val n = params.length

  def predict(x: Vector[Double]) = {
    (0 until n).map {
      case 0 => params(0)
      case i => params(i) * x(i-1)
    }.sum
  }

  /**
   * @param examples the (training) data set - list of (input, output) pairs
   * @return a tuple of (cost, derivatives)
   */
  def cost(examples: List[(Vector[Double], Double)], lambda: Double = 0): (Double, Vector[Double])= {
    // the errors for each example
    val errs = examples.par.map { case (xs, y) =>
      predict(xs) - y
    }.toList
    // num examples
    val m: Int = examples.size

    val penalty = 0 // TODO regularization: lambda * params.map(x => x*x).sum / (2d*m)
    // the cost - sum of squared errors
    val j = errs.par.map { err =>
      err * err
    }.sum / (2d*m) + penalty

    // the partial derivatives for each weight
    val derivs = (0 until n).par.map { i =>
      (examples,errs).zipped.map((ex,err) =>
        if (i == 0) err
        else ex._1(i-1) * err
      ).sum / m
    }.toVector
    (j, derivs)
  }
}

/**
 * Created by ian on 12/16/14.
 *
 * @author Ian Tabolt <iantabolt@gmail.com>
 */
object LinearModel {
  type CostFunction = (Vector[Double]) => (Double, Vector[Double])
  def gradientDescentMinimize(cf: CostFunction, initGuess: Vector[Double], stopCond: Double, alpha: Double, maxIterations: Int) = {
    // returns a stream of guesses tupled with the cost of that guess
    def guessStream(currParams: Vector[Double], maxIter: Int): Stream[(Vector[Double],Double)] = {
      if (maxIter == 0) Stream.empty
      else {
        val (j, derivs) = cf(currParams)
        print(f"\rCost: $j%8.6f")
        (currParams, j) #:: {
          val updated =
            (currParams, derivs).zipped.map { (param, dparam) =>
              param - alpha * dparam
            }
          guessStream(updated, maxIter - 1)
        }
      }
    }
    // adjacentGuesses is a stream of 2-element seqs holding adjacent values of guessStream(initGuess)
    // eg Seq((prevGuess,prevCost), (currGuess,currCost)) #:: ?
    val adjacentGuesses: Iterator[Stream[(Vector[Double], Double)]] =
      guessStream(initGuess, maxIterations).sliding(2)
    val maybeConverged: Option[Stream[(Vector[Double], Double)]] =
      adjacentGuesses.find { case Seq((_,cost0), (_,cost1)) => cost0 - cost1 < stopCond }
    println()
    maybeConverged.map { case Seq((params, _), _) => params }
        .getOrElse { throw new Error("Failed to converge after " + maxIterations + " iterations") }
  }

  def costFunction(dataset: List[(Vector[Double],Double)], lambda: Double): CostFunction = { params: Vector[Double] =>
    LinearModel(params).cost(dataset, lambda)
  }

  /**
   * Trains simple multi-variate linear regression via gradient descent
   * @param dataset a list of (x,y) pairs training data
   * @param initGuess the starting parameters (if empty, small random numbers will be used)
   * @param stopCond the difference in cost of two iterations that triggers the algorithm to stop
   * @param alpha the learning rate
   * @param maxIterations the maximum number of iterations for the algorithm to run
   * @return a Regression object with the optimum parameters
   */
  def train(dataset: List[(Vector[Double], Double)], initGuess: Option[Vector[Double]]=None,
            stopCond: Double=0.00001, alpha:Double=0.00007, lambda:Double=1.0, maxIterations: Int = 100000) = {
    val cf = costFunction(dataset, lambda)
    val optParams = gradientDescentMinimize(
      cf = cf,
      initGuess = initGuess.getOrElse(Vector.fill(dataset.head._1.length+1)(Random.nextGaussian()*0.2)),
      stopCond = stopCond,
      alpha = alpha,
      maxIterations = maxIterations)
    LinearModel(optParams)
  }

  def demoSimpleDataset() = {
    val data = List(
      Vector(1.0)->13.3,
      Vector(3.0)->16.1,
      Vector(6.0)->20.5
    )
    val optParams = train(data).params
    println(optParams)
    // should around theta0=11.8263, theta1=1.44210
  }

  def demoWineDataset() = {
    // returns a 3-tuple of (names, train (80%), test (20%))
    def readWineDataset(url: URL): (Vector[String], List[(Vector[Double], Double)], List[(Vector[Double], Double)]) = {
      val lines: Iterator[String] = io.Source.fromURL(url).getLines()
      val names = lines.next().split(";").toVector
      val dataset = lines.map { line =>
        val xs :+ y = line.split(";").toVector.map(_.toDouble)
        (xs, y)
      }.toList
      var i = 0
      val (train,test) = dataset.partition { _ =>
        i += 1
        i % 5 != 0
      }
      (names, train, test)
    }
    def eval(trainSet: List[(Vector[Double], Double)],
             testSet: List[(Vector[Double], Double)],
             names:Vector[String],
             preprocess: Vector[Double]=>Vector[Double],
             alpha: Double,
             stopCond: Double,
             processOutput: Double=>Double): Double = {
      def pp(dataset: List[(Vector[Double],Double)]) = {
        dataset.map { case (xs,y) =>
          (preprocess(xs),y)
        }
      }
      val tr = pp(trainSet)
      val te = pp(testSet)
      val model = train(tr, alpha=alpha, stopCond=stopCond)
      println("Learned params: ")
      model.params.drop(1).zip(names.dropRight(1))
        .sortBy(pn => math.abs(pn._1)).reverse
        .foreach{ case (param,name) =>
        println("  " + name + "\t" + param)
      }
      println("Train set cost: " + model.cost(tr)._1)
      println("Test set cost:  " + model.cost(te)._1)
      def countCorrect(data: List[(Vector[Double],Double)]) = {
        data.count { case (xs,y) =>
          y == processOutput(model.predict(xs)) }
      }.toDouble
      val trCorrect = countCorrect(tr)
      println(s"Train success rate: $trCorrect/${tr.size} (${trCorrect/tr.size})" )
      val teCorrect = countCorrect(te)
      println(s"Test success rate: $teCorrect/${te.size} (${teCorrect/te.size})" )
      model.cost(te)._1
    }
    def getOutput(prediction: Double) = {
      if (prediction < 0) 0d
      else if (prediction > 10) 10d
      else prediction.round.toDouble
    }

    val (whiteNames, whiteTrain, whiteTest) = readWineDataset(getClass.getResource("winequality-white.csv"))
    println("White wine:")
    val whiteNormalizer = FeatureNormalizer.fromTrainData(whiteTrain.map(_._1))
    println("Using normalizer: " + whiteNormalizer)
    eval(whiteTrain, whiteTest, whiteNames, whiteNormalizer, 1.0, 0.000001, getOutput)
    val (redNames, redTrain, redTest) = readWineDataset(getClass.getResource("winequality-red.csv"))
    println("Red wine:")
    val redNormalizer = FeatureNormalizer.fromTrainData(redTrain.map(_._1))
    println("Using normalizer: " + redNormalizer)
    eval(redTrain, redTest, redNames, redNormalizer, 1.0, 0.000001, getOutput)
  }

  def main(args: Array[String]): Unit = {
    demoWineDataset()
  }
}