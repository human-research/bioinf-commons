package org.jetbrains.bio.genome

import com.google.common.base.Joiner
import com.google.common.cache.Cache
import com.google.common.cache.CacheBuilder
import com.google.common.collect.ImmutableListMultimap
import com.google.common.collect.LinkedListMultimap
import com.google.common.collect.ListMultimap
import org.apache.commons.csv.CSVFormat
import org.jetbrains.bio.util.*
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/**
 * A shortcut for annotation caches.
 *
 * The cache uses soft-values policy, thus make sure the cached class
 * implements [.equals] and [.hashCode] either manually or via data
 * syntax sugar. Otherwise, there be heisenbugs!
 *
 * Based on a true story. Consider a long-living map `M` indexed by
 * genes. Once the memory is low, GC collects soft-referenced
 * 'ListMultimap<Chromosome, Gene>>', so the next cache access
 * re-populates the cache creating new instance for each of the genes.
 * If the 'Gene' relies on the default implementation of [.hashCode]
 * (which compares memory addresses), then `gene in M` would always
 * return false, and the entries of `M` could never be updated or
 * retrieved.
 */
internal fun <T> cache(): Cache<String, ListMultimap<Chromosome, T>> {
    return CacheBuilder.newBuilder()
            .softValues()
            .initialCapacity(1)
            .build<String, ListMultimap<Chromosome, T>>()
}

/** UCSC annotations always use "chrN" format, so we don't need [ChromosomeNamesMap]. */
internal fun chromosomeMap(build: String): Map<String, Chromosome> {
    return Genome[build].chromosomes.associateBy { it.name }
}

data class Repeat(val name: String,
                  override val location: Location,
                  val repeatClass: String,
                  val family: String) : LocationAware

/**
 * A registry for repetitive genomic elements.
 */
object Repeats {
    private val CACHE = cache<Repeat>()
    internal const val FILE_NAME = "rmsk.txt.gz"

    fun repeatsPath(genome: Genome): Path {
        val build = genome.build
        val repeatsPath = genome.dataPath / FILE_NAME
        repeatsPath.checkOrRecalculate("Repeats") { output ->
            val config = AnnotationsConfig[build]

            if (config.ucscAnnLegacyFormat) {
                // Builds with per-chromosome repeat annotations.
                val prefix = config.repeatsUrl.substringBeforeLast("/")
                val template = "%s_${config.repeatsUrl.substringAfterLast("/")}"
                UCSC.downloadBatchTo(output.path, build, "$prefix/", template)
            } else {
                config.repeatsUrl.downloadTo(output.path)
            }
        }
        return repeatsPath
    }

    private val FORMAT = CSVFormat.TDF.withHeader(
            "bin", "sw_score", "mismatches", "deletions", "insertions",
            "chrom", "genomic_start", "genomic_end", "genomic_left",
            "strand", "name", "class", "family", "repeat_start",
            "repeat_end", "repeat_left", "id")

    internal fun all(genome: Genome): ListMultimap<Chromosome, Repeat> {
        val build = genome.build
        return CACHE.get(build) {
            read(build, repeatsPath(genome))
        }
    }


    private fun read(build: String, repeatsPath: Path): ListMultimap<Chromosome, Repeat> {
        val builder = ImmutableListMultimap.builder<Chromosome, Repeat>()
        val chromosomes = chromosomeMap(build)
        FORMAT.parse(repeatsPath.bufferedReader()).use { csvParser ->
            for (row in csvParser) {
                val chromosome = chromosomes[row["chrom"]] ?: continue
                val strand = row["strand"].toStrand()
                val startOffset = row["genomic_start"].toInt()
                val endOffset = row["genomic_end"].toInt()
                val location = Location(startOffset, endOffset, chromosome, strand)
                val repeat = Repeat(row["name"], location,
                        row["class"].toLowerCase(),
                        row["family"].toLowerCase())
                builder.put(chromosome, repeat)
            }
        }

        return builder.build()
    }
}

data class CytoBand(val name: String,
                    val gieStain: String,
                    override val location: Location) :
        Comparable<CytoBand>, LocationAware {

    override fun compareTo(other: CytoBand) = location.compareTo(other.location)

    override fun toString() = "$name: $location"
}

/**
 * Chromosomes, bands and all that.
 */
object CytoBands {
    private val CACHE = cache<CytoBand>()

    internal const val FILE_NAME = "cytoBand.txt.gz"

    internal val FORMAT = CSVFormat.TDF
            .withHeader("chrom", "start_offset", "end_offset", "name", "gie_stain")

    internal fun all(genome: Genome): ListMultimap<Chromosome, CytoBand> {
        val build = genome.build
        return CACHE.get(build) {
            val cytobandsUrl = AnnotationsConfig[build].cytobandsUrl
            if (cytobandsUrl == null) {
                LinkedListMultimap.create()
            } else {
                val bandsPath = genome.dataPath / FILE_NAME
                bandsPath.checkOrRecalculate("CytoBands") { output ->
                    cytobandsUrl.downloadTo(output.path)
                }
                read(build, bandsPath)
            }
        }
    }

    private fun read(build: String, bandsPath: Path): ListMultimap<Chromosome, CytoBand> {
        val builder = ImmutableListMultimap.builder<Chromosome, CytoBand>()
        val chromosomes = chromosomeMap(build)
        FORMAT.parse(bandsPath.bufferedReader()).use { csvParser ->
            for (row in csvParser) {
                val chromosome = chromosomes[row["chrom"]] ?: continue

                val location = Location(row["start_offset"].toInt(), row["end_offset"].toInt(), chromosome)
                builder.put(chromosome, CytoBand(row["name"], row["gie_stain"], location))
            }
        }

        return builder.build()
    }
}

/**
 * A container for centromeres and telomeres.
 *
 * @author Roman Chernyatchik
 */
class Gap(val name: String, override val location: Location) :
        Comparable<Gap>, LocationAware {

    val isCentromere: Boolean get() = name == "centromere"

    val isTelomere: Boolean get() = name == "telomere"

    val isHeterochromatin: Boolean get() = name == "heterochromatin"

    override fun compareTo(other: Gap) = location.compareTo(other.location)
}

object Gaps {
    private val CACHE = cache<Gap>()

    internal const val FILE_NAME = "gap.txt.gz"

    internal val FORMAT = CSVFormat.TDF.withHeader(
            "bin", "chrom", "start_offset", "end_offset", "ix", "n",
            "size", "type", "bridge")

    private const val CENTROMERES_FILE_NAME = "centromeres.txt.gz"

    internal fun all(genome: Genome): ListMultimap<Chromosome, Gap> {
        val build = genome.build
        return CACHE.get(build) {
            val gapsPath = genome.dataPath / FILE_NAME
            gapsPath.checkOrRecalculate("Gaps") { output ->
                download(build, output.path)
            }

            read(build, gapsPath)
        }
    }

    private fun read(build: String, gapsPath: Path): ListMultimap<Chromosome, Gap> {
        val builder = ImmutableListMultimap.builder<Chromosome, Gap>()
        val chromosomes = chromosomeMap(build)
        FORMAT.parse(gapsPath.bufferedReader()).use { csvParser ->
            for (row in csvParser) {
                val chromosome = chromosomes[row["chrom"]] ?: continue
                val location = Location(
                        row["start_offset"].toInt(), row["end_offset"].toInt(),
                        chromosome, Strand.PLUS)
                builder.put(chromosome, Gap(row["type"].toLowerCase(), location))
            }
        }

        return builder.build()
    }

    private fun download(build: String, gapsPath: Path) {
        val config = AnnotationsConfig[build]

        if (config.ucscAnnLegacyFormat) {
            // Builds with per-chromosome gap annotations.
            val prefix = config.gapsUrl.substringBeforeLast("/")
            val template = "%s_${config.gapsUrl.substringAfterLast("/")}"
            UCSC.downloadBatchTo(gapsPath, build, "$prefix/", template)
        } else {
            config.gapsUrl.downloadTo(gapsPath)

            // Builds with separate centromere annotations:
            if (config.centromeresUrl != null) {
                // hg38 is special, it has centromere annotations in a
                // separate file. Obviously, the format of the file doesn't
                // match the one of 'gap.txt.gz', so we fake the left
                // out columns below.
                val centromeresPath = gapsPath.parent / CENTROMERES_FILE_NAME
                try {
                    config.centromeresUrl.downloadTo(centromeresPath)
                    centromeresPath.bufferedReader().useLines { centromeres ->
                        gapsPath.bufferedWriter(StandardOpenOption.WRITE,
                                StandardOpenOption.APPEND).use { target ->
                            for (line in centromeres) {
                                val leftout = Joiner.on('\t')
                                        .join("", -1, -1, "centromere", "no")
                                target.write(line.trimEnd() + leftout + '\n')
                            }
                            target.close()
                        }
                    }
                } catch (e: Exception) {
                    gapsPath.delete()  // we aren't done yet.
                } finally {
                    centromeresPath.deleteIfExists()
                }
            }
        }
    }
}

/**
 * A CpG island. Nobody knows what it is really.
 *
 * @author Roman Chernyatchik
 */
data class CpGIsland(
        /** Number of CpG dinucleotides within the island.  */
        val CpGNumber: Int,
        /** Number of C and G nucleotides within the island.  */
        val GCNumber: Int,
        /**
         * Ratio of observed CpG to expected CpG counts within the island,
         * where the expected number of CpGs is calculated as
         * `numC * numG / length`.
         */
        val observedToExpectedRatio: Double,
        override val location: Location) : LocationAware, Comparable<CpGIsland> {

    override fun compareTo(other: CpGIsland) = location.compareTo(other.location)
}

object CpGIslands {
    private val CACHE = cache<CpGIsland>()

    internal const val ISLANDS_FILE_NAME = "cpgIslandExt.txt.gz"

    fun cpgIslandsPath(genome: Genome): Path? {
        val path = genome.dataPath / ISLANDS_FILE_NAME
        if (genome.build == Genome.TEST_ORGANISM_BUILD) {
            return path
        }
        val url = AnnotationsConfig[genome.build].cpgIslandsUrl
        if (url != null) {
            path.checkOrRecalculate("CpGIslands") { output ->
                url.downloadTo(output.path)
            }
            return path
        }
        return null
    }

    internal fun all(genome: Genome): ListMultimap<Chromosome, CpGIsland> {
        val build = genome.build
        return CACHE.get(build) {
            val path = cpgIslandsPath(genome)
            if (path == null) {
                LinkedListMultimap.create()
            } else {
                read(build, path)
            }
        }
    }

    private fun read(build: String, islandsPath: Path): ListMultimap<Chromosome, CpGIsland> {
        val builder = ImmutableListMultimap.builder<Chromosome, CpGIsland>()
        val chromosomes = chromosomeMap(build)

        val headers = arrayOf("chrom", "start_offset", "end_offset", "name", "length",
                "cpg_num", "gc_num", "per_cpg", "per_gc", "obs_exp")

        val csvFormat = if (AnnotationsConfig[build].ucscAnnLegacyFormat) {
            //Builds missing `bin` column in the annotations
            CSVFormat.TDF.withHeader(*headers)
        } else {
            CSVFormat.TDF.withHeader("bin", *headers)
        }

        csvFormat.parse(islandsPath.bufferedReader()).use { csvParser ->
            for (row in csvParser) {
                val chromosome = chromosomes[row["chrom"]] ?: continue
                val location = Location(
                        row["start_offset"].toInt(), row["end_offset"].toInt(),
                        chromosome)
                val island = CpGIsland(
                        row["cpg_num"].toInt(), row["gc_num"].toInt(),
                        row["obs_exp"].toDouble(), location)
                builder.put(chromosome, island)
            }
        }

        return builder.build()
    }
}