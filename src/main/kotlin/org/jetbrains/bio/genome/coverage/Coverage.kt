package org.jetbrains.bio.genome.coverage

import com.google.common.annotations.VisibleForTesting
import gnu.trove.list.TIntList
import kotlinx.support.jdk7.use
import org.jetbrains.bio.genome.ChromosomeRange
import org.jetbrains.bio.genome.GenomeQuery
import org.jetbrains.bio.genome.Location
import org.jetbrains.bio.genome.Strand
import org.jetbrains.bio.npy.NpzFile
import java.io.IOException
import java.nio.file.Path


/**
 * [Coverage] allows to query the tags coverage of a given genomic [Location].
 * Currently, it has two implementations: [SingleEndCoverage] and [PairedEndCoverage].
 */
interface Coverage {

    val genomeQuery: GenomeQuery
    /**
     * Returns the number of tags inside a given [location].
     */
    fun getCoverage(location: Location): Int

    /**
     * Returns the number of tags inside a given [chromosomeRange]
     * (i.e. on both strands).
     */
    fun getBothStrandsCoverage(chromosomeRange: ChromosomeRange): Int =
            getCoverage(chromosomeRange.on(Strand.PLUS)) +
                    getCoverage(chromosomeRange.on(Strand.MINUS))

    val depth: Long

    companion object {

        /**
         * Binary storage format version. Loader will throw an [IOException]
         * if it doesn't match.
         */
        const val VERSION = 4
        const val VERSION_FIELD = "version"
        const val PAIRED_FIELD = "paired"

        @Throws(IOException::class)
        internal fun load(
                inputPath: Path,
                genomeQuery: GenomeQuery,
                fragment: Fragment = AutoFragment
        ): Coverage {
            return NpzFile.read(inputPath).use { reader ->
                val version = reader[VERSION_FIELD].asIntArray().single()
                check(version == VERSION) {
                    "$inputPath coverage version is $version instead of $VERSION"
                }

                val paired = reader[PAIRED_FIELD].asBooleanArray().single()
                if (paired) {
                    PairedEndCoverage.load(reader, genomeQuery)
                } else {
                    SingleEndCoverage.load(reader, genomeQuery).withFragment(fragment)
                }
            }
        }
    }
}


sealed class Fragment {

    /**
     * At some point, fragment was stored as Int?, with null corresponding to what now is [AutoFragment].
     * Since some parts of code (like ID generation) depend on this convention, this is a useful conversion property.
     */
    abstract val nullableInt: Int?

    companion object {
        @Throws(NumberFormatException::class)
        fun fromString(value: String): Fragment {
            if (value == "auto") {
                return AutoFragment
            }
            return FixedFragment(Integer.parseInt(value))
        }
    }
}

data class FixedFragment(val size: Int) : Fragment() {
    override fun toString(): String = size.toString()
    override val nullableInt: Int? = size
}

object AutoFragment : Fragment() {
    override fun toString(): String = "auto"
    override val nullableInt: Int? = null
}

/**
 * Returns the insertion index of [target] into a sorted list.
 *
 * If [target] already appears in this vector, the returned
 * index is just before the leftmost occurrence of [target].
 */
@VisibleForTesting
internal fun TIntList.binarySearchLeft(target: Int): Int {
    var lo = 0
    var hi = size()
    while (lo < hi) {
        val mid = (lo + hi) ushr 1
        if (target <= this[mid]) {
            hi = mid
        } else {
            lo = mid + 1
        }
    }

    return lo
}