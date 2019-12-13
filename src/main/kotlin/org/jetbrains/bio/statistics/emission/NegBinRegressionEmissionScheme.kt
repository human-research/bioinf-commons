package org.jetbrains.bio.statistics.emission

import org.apache.commons.math3.special.Gamma
import org.apache.commons.math3.util.FastMath
import org.jetbrains.bio.dataframe.DataFrame
import org.jetbrains.bio.statistics.MoreMath
import org.jetbrains.bio.statistics.distribution.NegativeBinomialDistribution
import org.jetbrains.bio.statistics.distribution.Sampling
import org.jetbrains.bio.viktor.F64Array
import org.jetbrains.bio.viktor.asF64Array
import java.util.function.IntPredicate
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

/**
 *
 * Negative Binomial regression.
 *
 * @author Elena Kartysheva
 * @date 9/13/19
 */
class NegBinRegressionEmissionScheme(
        covariateLabels: List<String>,
        regressionCoefficients: DoubleArray,
        failures: Double
) : IntegerRegressionEmissionScheme(covariateLabels, regressionCoefficients) {

    var failures = failures
        private set
    var fLogf = failures*ln(failures)
        private set
    override fun mean(eta: Double) = exp(eta)
    override fun meanDerivative(eta: Double) = exp(eta)
    override fun meanVariance(mean: Double) = mean + mean*mean/failures

    override fun sampler(mean: Double) = Sampling.sampleNegBinomial(mean, failures)

    override fun meanInPlace(eta: F64Array) = eta.apply { expInPlace() }
    override fun meanDerivativeInPlace(eta: F64Array) = eta.apply { expInPlace() }
    override fun meanVarianceInPlace(mean: F64Array) = mean + mean*mean/failures

    override fun zW(y: F64Array, eta: F64Array): Pair<F64Array, F64Array> {
        // Since h(η) = h'(η) = var(h(η)), we can skip h'(η) and var(h(η)) calculations and simplify W:
        // W = diag(h'(η)^2 / var(h(η))) = h(η)
        val countedLink = meanInPlace(eta.copy())
        eta += (y - countedLink).apply { divAssign(countedLink) }
        return eta to countedLink
    }

    override fun logProbability(df: DataFrame, t: Int, d: Int): Double {
        // We don't use the existing Poisson log probability because that saves us one logarithm.
        // We would have to provide lambda = exp(logLambda), and the Poisson implementation would then have to
        // calculate log(lambda) again.
        val mean = getPredictor(df, t)
        val y = df.getAsInt(t, df.labels[d])
        val logMeanPlusFailure = ln(mean + failures)

        return when {
            failures.isNaN() || y < 0 || y == Integer.MAX_VALUE -> Double.NEGATIVE_INFINITY
            mean == 0.0 -> if (y == 0) 0.0 else Double.NEGATIVE_INFINITY
            failures.isInfinite() -> y * ln(y.toFloat()) - mean - MoreMath.factorialLog(y)
            else -> {
                Gamma.logGamma(y + failures) - MoreMath.factorialLog(y) + fLogf - failures*logMeanPlusFailure +
                        y*(ln(mean) - logMeanPlusFailure)
            }
        }
    }

    override fun update(df: DataFrame, d: Int, weights: F64Array) {
        val X = generateDesignMatrix(df)
        val yInt = df.sliceAsInt(df.labels[d])
        val y = DoubleArray (yInt.size) {yInt[it].toDouble()}.asF64Array()
        val iterMax = 100
        val tol = 1e-8
        var beta0 = regressionCoefficients
        var beta1 = regressionCoefficients
        for (i in 0 until iterMax) {
            val eta = WLSRegression.calculateEta(X, beta0)
            failures = NegativeBinomialDistribution.fitNumberOfFailures(y, weights, meanInPlace(eta.copy()), failures)
            val (z, W) = zW(y, eta)
            W *= weights
            beta1 = WLSRegression.calculateBeta(X, z, W)
            if ((beta1.zip(beta0) { a, b -> abs(a - b) }).sum() < tol) {
                break
            }
            beta0 = beta1
        }
        regressionCoefficients = beta1
        fLogf = failures*ln(failures)
    }
}


fun main(args: Array<String>) {
    val covar = DataFrame()
            .with("y", IntArray(1000000))
            .with("x1", DoubleArray(1000000) { Random.nextDouble(0.0, 1.0) })
            .with("x2", DoubleArray(1000000) { Random.nextDouble(0.0, 1.0) })

    val regrES = NegBinRegressionEmissionScheme(listOf("x1", "x2"), doubleArrayOf(1.0, -2.0, 3.0), 7.0)
    val pred = IntPredicate {true}

    regrES.sample(covar, 0, pred)
    val regrES2 = NegBinRegressionEmissionScheme(listOf("x1", "x2"), doubleArrayOf(0.5, 0.5, 0.5), 4.0)
    println("Update")
    regrES2.update(covar, 0, DoubleArray(1000000, {1.0}).asF64Array())
    println("Beta: ${regrES2.regressionCoefficients.asList()}")
    println("Failures: ${regrES2.failures}")
}