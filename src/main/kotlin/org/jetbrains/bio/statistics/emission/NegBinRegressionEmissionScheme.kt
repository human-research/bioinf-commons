package org.jetbrains.bio.statistics.emission

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
        regressionCoefficients: DoubleArray
) : IntegerRegressionEmissionScheme(covariateLabels, regressionCoefficients) {

    var failitures = 1.0
        private set
    override fun mean(eta: Double) = exp(eta)
    override fun meanDerivative(eta: Double) = exp(eta)
    override fun meanVariance(mean: Double) = mean + mean*mean/failitures

    override fun sampler(mean: Double) = Sampling.samplePoisson(mean)

    override fun meanInPlace(eta: F64Array) = eta.apply { expInPlace() }
    override fun meanDerivativeInPlace(eta: F64Array) = eta.apply { expInPlace() }
    override fun meanVarianceInPlace(mean: F64Array) = mean

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
        val logLambda = getPredictor(df, t)
        val y = df.getAsInt(t, df.labels[d])
        return y * logLambda - MoreMath.factorialLog(y) - FastMath.exp(logLambda)
    }

    override fun update(df: DataFrame, d: Int, weights: F64Array) {
        val X = generateDesignMatrix(df)
        val yInt = df.sliceAsInt(df.labels[d])
        failitures = NegativeBinomialDistribution.fitNumberOfFailures(yInt, weights, 1.0, failitures)
        val y = DoubleArray (yInt.size) {yInt[it].toDouble()}.asF64Array()
        val iterMax = 5
        val tol = 1e-8
        var beta0 = regressionCoefficients
        var beta1 = regressionCoefficients
        for (i in 0 until iterMax) {
            val eta = WLSRegression.calculateEta(X, beta0)
            val (z, W) = zW(y, eta)
            W *= weights
            beta1 = WLSRegression.calculateBeta(X, z, W)
            if ((beta1.zip(beta0) { a, b -> abs(a - b) }).sum() < tol) {
                break
            }
            beta0 = beta1
        }
        regressionCoefficients = beta1
    }
}