package org.jetbrains.bio.io

import kotlinx.support.jdk7.use
import org.jetbrains.bio.big.BedEntry
import org.jetbrains.bio.big.ExtendedBedEntry
import org.jetbrains.bio.genome.GenomeQuery
import org.jetbrains.bio.genome.Location
import org.jetbrains.bio.genome.Strand
import org.jetbrains.bio.util.*
import org.junit.Assert.assertArrayEquals
import org.junit.Rule
import org.junit.Test
import org.junit.rules.ExpectedException
import java.awt.Color
import java.io.IOException
import java.io.StringWriter
import java.nio.file.Path
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * See format details at https://genome.ucsc.edu/FAQ/FAQformat.html#format1
 */
class BedParserTest {
    @get:Rule
        var expectedEx = ExpectedException.none()

    private val CONTENT_RGB_EXAMPLE_SPACE_INSTEAD_OF_TABS =
            "chr2    127471196  127472363  Pos1  0  +  127471196  127472363  255,0,0\n" +
                    "chr2    127472363  127473530  Pos2  0  +  127472363  127473530  255,0,0\n" +
                    "chr2    127473530  127474697  Pos3  0  +  127473530  127474697  255,0,0\n" +
                    "chr2    127474697  127475864  Pos4  0  +  127474697  127475864  255,0,0\n" +
                    "chr2    127475864  127477031  Neg1  0  -  127475864  127477031  0,0,255"

    private val CONTENT_RGB_EXAMPLE =
            "chr2\t127471196\t127472363\tPos1\t0\t+\t127471196\t127472363\t255,0,0\n" +
                    "chr2\t127472363\t127473530\tPos2\t0\t+\t127472363\t127473530\t255,0,0\n" +
                    "chr2\t127473530\t127474697\tPos3\t0\t+\t127473530\t127474697\t255,0,0\n" +
                    "chr2\t127474697\t127475864\tPos4\t0\t+\t127474697\t127475864\t255,0,0\n" +
                    "chr2\t127475864\t127477031\tNeg1\t0\t-\t127475864\t127477031\t0,0,255"


    @Test
    fun testDefaultBedFormat_noRgb() {
        val contents = "chr1 1000 5000 cloneA 960 + 1000 5000 0 2 567,488, 0,3512\n" +
                "chr1 2000 6000 cloneB 900 - 2000 6000 0 2 433,399, 0,3601"

        val format = BedFormat.DEFAULT.delimiter(' ')

        withBedFile(contents) { path ->
            // Count
            assertEquals(2, format.parse(path) { it.count() })

            // Items
            val firstEntry = format.parse(path) { it.first() }
            assertEquals(ExtendedBedEntry("chr1", 1000, 5000, "cloneA",
                                          960, '+', 1000, 5000, 0, 2,
                                          intArrayOf(567, 488), intArrayOf(0, 3512)),
                         firstEntry.unpack(format))
        }
    }

    @Test
    fun testDefaultBedFormat_SecondEntry() {
        val contents = "chr1 1000 5000 cloneA 960 + 1000 5000\n" +
                "chr1 2000 6000 cloneB 900 - 2000 6000"

        val format = BedFormat.DEFAULT.delimiter(' ')

        withBedFile(contents) { path ->
            // Count
            assertEquals(2, format.parse(path) { it.count() })

            // Items
            format.parse(path) {
                val iterator = it.iterator()
                iterator.next()
                val secondEntry = iterator.next()
                assertEquals(ExtendedBedEntry("chr1", 2000, 6000, "cloneB",
                                              900, '-', 2000, 6000),
                             secondEntry.unpack(format))
            }
        }
    }

    @Test
    fun testDefaultBedFormat_WithRgb() {
        val format = BedFormat.DEFAULT
        withBedFile(CONTENT_RGB_EXAMPLE) { path ->
            // Count
            assertEquals(5, format.parse(path) { it.count() })

            // Items
            val firstEntry = format.parse(path) { it.first() }
            assertEquals(ExtendedBedEntry("chr2", 127471196, 127472363, "Pos1",
                                          0, '+', 127471196, 127472363,
                                          Color(255, 0, 0).rgb),
                         firstEntry.unpack(BedFormat.from("bed9")))
        }
    }


    @Test
    fun testSpacesInsteadOfTabs() {
        val format = BedFormat.DEFAULT.delimiter(' ')
        withBedFile(CONTENT_RGB_EXAMPLE_SPACE_INSTEAD_OF_TABS) { path ->
            // Count
            assertEquals(5, format.parse(path) { it.count() })

            // Items
            val firstEntry = format.parse(path) { it.first() }
            assertEquals(ExtendedBedEntry("chr2", 127471196, 127472363, "Pos1",
                                          0, '+', 127471196, 127472363,
                                          Color(255, 0, 0).rgb),
                         firstEntry.unpack(BedFormat.from("bed9",' '), omitEmptyStrings = true))
        }
    }

    @Test
    fun testDefaultBedFormat_Separators() {
        val tabs = "chr2\t    127471196  \t  127472363\tPos1\t0\t+\t127471196\t127472363\t255,0,0"
        val mixed = "chr2\t127474697\t127475864\tPos4 \t0\t+\t127474697\t127475864\t255,0,0"
        val contents = tabs + "\n" + mixed

        val format = BedFormat.DEFAULT
        withBedFile(contents) { path ->

            // Count
            assertEquals(2, format.parse(path) { it.count() })

            // Items
            format.parse(path) {
                for (entry in it) {
                    assertEquals("chr2", entry.chrom)
                }
            }
        }
    }

    @Test
    fun testBedWithSkippedRecords_SkipOneRecord() {
        // XXX skipping not supported

        val skipped = "chr1 1000 5000 cloneA + 1000 5000 0 2 567,488, 0,3512\n" +
                "chr1 2000 6000 cloneB - 2000 6000 0 2 433,399, 0,3601"

        withBedFile(skipped) { path ->
            val bedFormat = BedFormat.auto(path)
            // Count
            assertEquals(2, bedFormat.parse(path) { it.count() })

            // Items
            val firstEntry = bedFormat.parse(path) { it.first() }
            assertEquals(
                    ExtendedBedEntry(
                            "chr1", 1000, 5000, "cloneA", 0,
                            extraFields = "+\t1000\t5000\t0\t2\t567,488,\t0,3512".split('\t').toTypedArray()
                    ),
                    firstEntry.unpack(bedFormat))
        }
    }

    @Test
    fun testLastSymbolSplit() {
        withBedFile("chr1 1000 2000 +") { path ->
            val bedFormat = BedFormat.from("bed3+1", ' ')
            assertEquals(1, bedFormat.parse(path) { it.count() })
        }
    }

    @Test
    fun testBedWithSkippedRecords_SkipSeveralRecords() {
        // XXX: Not supported:

        val skipped = "chr1 1 1000 cloneA + 1000 5000 2 0,3512\n" +
                "chr1 1 2000 cloneB - 2000 6000 2 0,3601"

        withBedFile(skipped) { path ->
            assertEquals(BedFormat.from("bed4+5", ' '), BedFormat.auto(path))
        }
    }

    @Test
    fun testBedWithSkippedRecords_SkipSeveralRecords_AtOnce() {
        // XXX: Not supported:

        val skipped = "chr1 1 1000 + 1000 5000 2 0,3512\n" +
                "chr1 1 2000 - 2000 6000 2 0,3601"

        withBedFile(skipped) { path ->
            assertEquals(BedFormat.from("bed3+5", ' '), BedFormat.auto(path))
        }
    }

    @Test
    fun testBedSkippedObligatoryFields() {

        val content = "chr1 1000 cloneA + 1000 5000 2 0,3512\n" +
                "chr1 2000 cloneB - 2000 6000 2 0,3601"

        withBedFile(content) { path ->
            expectedEx.expect(IllegalArgumentException::class.java)
            expectedEx.expectMessage("""Source: $path
Unknown BED format:
chr1 1000 cloneA + 1000 5000 2 0,3512
Fields number in BED file is between 3 and 15, but was 2""")

            BedFormat.auto(path)
        }
    }

    @Test
    fun testComment() {
        val contents = "# comment\nchr1 1000 2000"

        withBedFile(contents) { path ->
            val bedFormat = BedFormat.from("bed3",' ')
            assertEquals(1, bedFormat.parse(path) { it.count() })
        }
    }

    @Test
    fun testTrackNameComment() {
        val contents = "track name=\"FooBar\"\n" +
                "chr1 1000 5000"

        withBedFile(contents) { path ->
            val bedFormat = BedFormat.from("bed3",' ')
            assertEquals(1, bedFormat.parse(path) { it.count() })
        }
    }

    @Test
    fun testWriteBedWithSkippedRecords() {
        // XXX: skipping records not supported
        val bedFormat = BedFormat.from("bed8+1")

        val writer = StringWriter()
        bedFormat.print(writer).use { bedPrinter ->
            bedPrinter.print(ExtendedBedEntry("chr1", 1000, 5000, "cloneA",
                                              777, '-', 2000, 4000, Color.WHITE.rgb, 2,
                                              intArrayOf(567, 488), intArrayOf(0, 3512), arrayOf("f1", "f2")))
        }

        assertEquals(
                "chr1\t1000\t5000\tcloneA\t777\t-\t2000\t4000\tf1\n",
                writer.toString().replace("\r", ""))
    }

    @Test
    fun testWriteBed() {
        val bedFormat = BedFormat.DEFAULT
        val writer = StringWriter()
        bedFormat.print(writer).use { bedPrinter ->
            bedPrinter.print(ExtendedBedEntry("chr1", 1000, 5000, "cloneA",
                                              777, '-', 1000, 5000, Color.BLACK.rgb, 2,
                                              intArrayOf(567, 488), intArrayOf(0, 3512)))
        }

        assertEquals(
                "chr1\t1000\t5000\tcloneA\t777\t-\t1000\t5000\t0,0,0\t2\t567,488\t0,3512\n",
                writer.toString().replace("\r", ""))
    }

    @Test
    fun testWriteBed6() {
        val bedFormat = BedFormat.from("bed6")
        val writer = StringWriter()
        bedFormat.print(writer).use { bedPrinter ->
            bedPrinter.print(ExtendedBedEntry("chr1", 1000, 5000, "cloneA",
                                              777, '-', 1000, 5000, Color.BLACK.rgb, 2,
                                              intArrayOf(567, 488), intArrayOf(0, 3512)))
        }

        assertEquals(
                "chr1\t1000\t5000\tcloneA\t777\t-\n",
                writer.toString().replace("\r", ""))
    }

    @Test
    fun testWriteBedRGB() {
        val bedFormat = BedFormat.RGB
        val writer = StringWriter()
        bedFormat.print(writer).use { bedPrinter ->
            bedPrinter.print(ExtendedBedEntry("chr1", 1000, 5000, "cloneA",
                                              777, '-', 1000, 5000, Color.BLACK.rgb, 2,
                                              intArrayOf(567, 488), intArrayOf(0, 3512)))
        }

        assertEquals(
                "chr1\t1000\t5000\tcloneA\t777\t-\t1000\t5000\t0,0,0\n",
                writer.toString().replace("\r", ""))
    }

    @Test
    fun testAuto_FromText() {
        assertEquals(
                BedFormat(12, 0),
                BedFormat.auto("chr2\t1\t2\tDescription\t0\t+\t1000\t5000\t255,0,0\t2\t10,20\t1,2", null)
        )
        assertEquals(
                BedFormat(12, 0, ' '),
                BedFormat.auto("chr2 1 2 Description 0 + 1000 5000 255,0,0 2 10,20 1,2", null)
        )
    }

    @Test
    fun testAuto_Default() {
        doCheckAuto("chr2\t1\t2\tDescription\t0\t+\t1000\t5000\t255,0,0\t2\t10,20\t1,2",
                    '\t', "bed12",
                    ExtendedBedEntry("chr2", 1, 2, "Description",
                                      0, '+', 1000, 5000, Color.RED.rgb,
                                      2, intArrayOf(10, 20), intArrayOf(1, 2)))
    }

    @Test
    fun testAuto_FromBed() {
        withResource(BedParserTest::class.java, "bed12.bed") { path ->
            assertEquals("(bed12, '\t')", BedFormat.auto(path).toString())
        }
    }

    @Test
    fun testAuto_FromBedZip() {
        withResource(BedParserTest::class.java, "bed12.bed.zip") { path ->
            assertEquals("(bed12, '\t')", BedFormat.auto(path).toString())
        }
    }

    @Test
    fun testAuto_FromBedGz() {
        withResource(BedParserTest::class.java, "bed12.bed.gz") { path ->
            assertEquals("(bed12, '\t')", BedFormat.auto(path).toString())
        }
    }

    @Test
    fun testParse_FromBed() {
        withResource(BedParserTest::class.java, "bed12.bed") { path ->
            val entry = BedFormat.auto(path).parse(path) { it.first() }
            assertEquals(BedEntry("chr2", 1, 2, "Description\t0\t+\t1000\t5000\t255,0,0\t2\t10,20\t1,2"), entry)
        }
    }

    @Test
    fun testParse_FromBedZip() {
        withResource(BedParserTest::class.java, "bed12.bed.zip") { path ->
            val entry = BedFormat.auto(path).parse(path) { it.first() }
            assertEquals(BedEntry("chr2", 1, 2, "Description\t0\t+\t1000\t5000\t255,0,0\t2\t10,20\t1,2"), entry)
        }
    }

    @Test
    fun testParse_FromBedGz() {
        withResource(BedParserTest::class.java, "bed12.bed.gz") { path ->
            val entry = BedFormat.auto(path).parse(path) { it.first() }
            assertEquals(BedEntry("chr2", 1, 2, "Description\t0\t+\t1000\t5000\t255,0,0\t2\t10,20\t1,2"), entry)
        }
    }

    @Test
    fun testAuto_WhitespaceSep() {
        doCheckAuto("chr2 1 2 Description 960 + 1000 5000 0 2 10,20, 1,2",
                    ' ', "bed12",
                    ExtendedBedEntry("chr2", 1, 2, "Description",
                                      960, '+', 1000, 5000, 0, 2,
                                      intArrayOf(10, 20), intArrayOf(1, 2)))
    }

    @Test
    fun testAuto_MinimalFormat() {
        // BED lines have three required fields and nine additional optional fields:

        doCheckAuto("chr2\t1\t2", '\t', "bed3",
                    ExtendedBedEntry("chr2", 1, 2))
    }

    @Test
    fun testAuto_SkippedStrandColumn() {
        // XXX: Not supported:

        doCheckAuto("chr2\t1\t2\tDescription\t0\t1000\t5000\t255\t2\t10,20\t1,2",
                    '\t', "bed5+6",
                    ExtendedBedEntry("chr2", 1, 2, "Description",
                                      extraFields = "1000\t5000\t255\t2\t10,20\t1,2".split('\t').toTypedArray())
        )
    }

    @Test
    fun testAuto_SkippedScoreAndTruncated() {
        // no score, quite popular modification, e.g. roadmap epigenomics chipseqs:
        //XXX: skipping fields not supported

        doCheckAuto("chr2\t1\t2\tDescription\t-",
                    '\t', "bed4+1",
                    ExtendedBedEntry("chr2", 1, 2, "Description",
                                      extraFields = arrayOf("-"))
        )
    }

    @Test
    fun testAuto_DoubleScore() {
        doCheckAuto("chr2\t1\t2\tDescription\t0.200\t-",
                    '\t', "bed4+2",
                    ExtendedBedEntry("chr2", 1, 2, "Description",
                                      extraFields = arrayOf("0.200", "-"))
        )
    }

    @Test
    fun testAuto_DoubleScore2() {
        doCheckAuto("chr2\t1\t2\tDescription\t0.200",
                    '\t', "bed4+1",
                    ExtendedBedEntry("chr2", 1, 2, "Description",
                                      extraFields = arrayOf("0.200"))
        )
    }

    @Test
    fun testAuto_SkippedSeveralColumns() {
        // Name and Score skipped
        //XXX: skipping fields not supported

        doCheckAuto("chr2\t1\t2\t+\t1000", '\t', "bed3+2",
                    ExtendedBedEntry("chr2", 1, 2,
                                      extraFields = arrayOf("+", "1000"))
        )
    }

    @Test
    fun testAuto_Simple() {
        doCheckAuto("chr1\t10051\t10250\tHWI-ST700693:250:D0TG9ACXX:3:2112:8798:84378\t1\t-\n",
                    '\t', "bed6",
                    ExtendedBedEntry("chr1", 10051, 10250,
                                      "HWI-ST700693:250:D0TG9ACXX:3:2112:8798:84378", 1, '-')
        )
    }

    @Test
    fun testAuto_SkippScore() {
        //XXX: skipping fields not supported

        doCheckAuto("chr1\t10051\t10250\tHWI-ST700693:250:D0TG9ACXX:3:2112:8798:84378\t-\n",
                    '\t', "bed4+1",
                    ExtendedBedEntry("chr1", 10051, 10250,
                                      "HWI-ST700693:250:D0TG9ACXX:3:2112:8798:84378", 0, '.',
                                      extraFields = arrayOf("-"))
        )
    }

    @Test
    fun testAuto_DefaultScheme() {
        withBedFile { path ->
            BedFormat().print(path.bufferedWriter()).use { bedPrinter ->
                val entry = ExtendedBedEntry("chr2", 1, 2, "Description",
                                                  0, '+', 1000, 5000, Color.RED.rgb)
                bedPrinter.print(entry)
            }

            doCheckAuto(path.read(), '\t', "bed6",
                        ExtendedBedEntry("chr2", 1, 2, "Description", strand = '+'))
        }
    }


    @Test
    fun testAutoMacs2Peaks_Extra3() {
        doCheckAuto("chr1\t713739\t714020\tout_peak_1\t152\t.\t6.84925\t17.97400\t173\n",
                    '\t', "bed6+3",
                    ExtendedBedEntry("chr1", 713739, 714020, "out_peak_1", 152, '.',
                                      extraFields = "6.84925\t17.97400\t173".split('\t').toTypedArray())
        )
    }

    @Test
    fun testAutoMacs2Peaks_Extra4() {
        doCheckAuto("chr1\t713739\t714020\tout_peak_1\t152\t.\t6.84925\t17.97400\t15.25668\t173\n",
                    '\t', "bed6+4",
                    ExtendedBedEntry("chr1", 713739, 714020, "out_peak_1", 152, '.',
                                      extraFields = "6.84925\t17.97400\t15.25668\t173".split('\t').toTypedArray())
        )
    }


    private fun doCheckAuto(contents: String, delimiter: Char, expectedFormat: String,
                            expectedEntry: ExtendedBedEntry) {
        withBedFile(contents) { path ->
            val format = BedFormat.auto(path)

            assertEquals(BedFormat.from(expectedFormat,delimiter), format)

            assertEquals(expectedEntry,
                         format.parse(path) {it.first().unpack(format)})
        }
    }

    @Test
    fun autoBedEntry() {
        assertEquals(BedFormat.from("bed3"), BedFormat.auto(BedEntry("chr1", 1, 3)))
        assertEquals(BedFormat.from("bed4"), BedFormat.auto(BedEntry("chr1", 1, 3, ".")))
        assertEquals(BedFormat.from("bed4"), BedFormat.auto(BedEntry("chr1", 1, 3, "foo")))

        assertEquals(BedFormat.from("bed6"), BedFormat.auto(BedEntry("chr1", 1, 3, ".\t1000\t+")))

        assertEquals(BedFormat.from("bed6", ' '),
                     BedFormat.auto(BedEntry("chr1", 1, 3, ". 1000 +")))

        assertEquals(BedFormat.from("bed6", ' '),
                     BedFormat.auto(BedEntry("chr1", 1, 3, ".  1000  +")))

        assertEquals(
                BedFormat.from("bed12"),
                BedFormat.auto(ExtendedBedEntry("chr1", 1, 3, "foo").pack())
        )

        assertEquals(
                BedFormat.from("bed12+2"),
                BedFormat.auto(ExtendedBedEntry("chr1", 1, 3, "foo",
                                                extraFields = arrayOf("f1", "f2")).pack())
        )

        assertEquals(
                BedFormat.from("bed12", ' '),
                BedFormat.auto(ExtendedBedEntry("chr1", 1, 3, "foo").pack(delimiter = ' '))
        )

        assertEquals(
                BedFormat.from("bed3"),
                BedFormat.auto(ExtendedBedEntry("chr1", 1, 3, "foo",
                                                extraFields = arrayOf("f1", "f2")).pack(3, 0))
        )

        assertEquals(
                BedFormat.from("bed4"),
                BedFormat.auto(ExtendedBedEntry("chr1", 1, 3, "foo",
                                                extraFields = arrayOf("f1", "f2")).pack(3, 1))
        )

        assertEquals(
                BedFormat.from("bed6"),
                BedFormat.auto(ExtendedBedEntry("chr1", 1, 3, "foo",
                                                extraFields = arrayOf("f1", "f2")).pack(6, 0))
        )

        assertEquals(
                BedFormat.from("bed6+2"),
                BedFormat.auto(ExtendedBedEntry("chr1", 1, 3, "foo",
                                                extraFields = arrayOf("f1", "f2")).pack(6))
        )
    }

    @Test
    fun parseFormatString() {
        assertEquals(6.toByte() to null, BedFormat.parseFormatString("bed6+"))
        assertEquals(6.toByte() to 3, BedFormat.parseFormatString("bed6+3"))
        assertEquals(3.toByte() to 0, BedFormat.parseFormatString("bed3"))
        assertEquals(15.toByte() to 0, BedFormat.parseFormatString("bed15"))

        assertEquals(2.toByte() to 0, BedFormat.parseFormatString("bed2"))
        assertEquals(16.toByte() to 0, BedFormat.parseFormatString("bed16"))
    }

    @Test(expected = IllegalStateException::class)
    fun parseFormatStringMalformated1() {
        BedFormat.parseFormatString("3+3")
    }

    @Test(expected = NumberFormatException::class)
    fun parseFormatStringMalformated2() {
        BedFormat.parseFormatString("bedX")
    }

    @Test(expected = NumberFormatException::class)
    fun parseFormatStringMalformated3() {
        BedFormat.parseFormatString("bed3p4")
    }

    @Test fun formatContains() {
        assertTrue(BedField.STRAND in BedFormat.from("bed6"))
        assertTrue(BedField.STRAND in BedFormat.from("bed7"))
        assertFalse(BedField.STRAND in BedFormat.from("bed5"))

        assertTrue(BedField.ITEM_RGB in BedFormat.from("bed9"))
        assertFalse(BedField.ITEM_RGB in BedFormat.from("bed8"))
    }

    @Test fun bedFormatFromFiled() {
        assertEquals("bed6", BedFormat(BedField.STRAND).fmtStr)
        assertEquals("bed9", BedFormat(BedField.ITEM_RGB).fmtStr)
    }

    @Test
    fun detectDelimiterInText() {
        assertEquals(
                '\t',
                BedFormat.detectDelimiter("chr1\t2\t3\nchr1\t2\t3\n", ';')
        )
        assertEquals(
                ' ',
                BedFormat.detectDelimiter("chr1 2 3\nchr1\t2\t3\n", ';')
        )
        assertEquals(
                '\t',
                BedFormat.detectDelimiter("#chr1 2 3\nchr1\t2\t3\n", ';')
        )
        assertEquals(
                ';',
                BedFormat.detectDelimiter("chr1,2,3\nchr1\t2\t3\n", ';')
        )
        assertEquals(
                ';',
                BedFormat.detectDelimiter("chr1#2#3\nchr1\t2\t3\n", ';')
        )
    }
    
    @Test
    fun detectDelimiter() {
        withFile(null, "chr1\t2\t3") {
            assertEquals('\t', BedFormat.detectDelimiter(it))
        }
        withFile("txt", "chr1\t2\t3") {
            assertEquals('\t', BedFormat.detectDelimiter(it))
        }
        withFile("tsv", "chr1\t2\t3") {
            assertEquals('\t', BedFormat.detectDelimiter(it))
        }

        withFile("tsv", "chr1\t2\t3") {
            assertEquals('\t', BedFormat.detectDelimiter(it.toUri()))
        }
    }

    @Test
    fun detectDelimiterBed() {
        withBedFile("chr1\t2\t3") {
            assertEquals('\t', BedFormat.detectDelimiter(it))
        }
        withBedFile("chr1,2,3") {
            assertEquals('\t', BedFormat.detectDelimiter(it))
        }
        withBedFile("chr1\t2,3") {
            assertEquals('\t', BedFormat.detectDelimiter(it))
        }
        withBedFile("chr1 2 3") {
            assertEquals(' ', BedFormat.detectDelimiter(it))
        }
        withBedFile("chr1\t2 3") {
            assertEquals('\t', BedFormat.detectDelimiter(it))
        }
    }

    @Test
    fun detectDelimiterCsv() {
        withFile("csv", "chr1,2\t3") {
            assertEquals(',', BedFormat.detectDelimiter(it))
        }
        withFile("csv", "chr1\t2\t3") {
            assertEquals('\t', BedFormat.detectDelimiter(it))
        }
        withFile("cSv", "chr1 2 3") {
            assertEquals(' ', BedFormat.detectDelimiter(it))
        }
    }

    @Test
    fun splitToInts() {
        assertArrayEquals(intArrayOf(1,2,3), "1,2,3".splitToInts(3))
        assertArrayEquals(intArrayOf(1,2,3), "1,2,3,4,5".splitToInts(3))

        assertArrayEquals(intArrayOf(1,2,3,4,5), "1,2,3,4,5".splitToInts(-1))
        assertArrayEquals(intArrayOf(1,2,0), "1,2,".splitToInts(-1))
        assertArrayEquals(intArrayOf(0,0,0), ",,".splitToInts(-1))

        assertArrayEquals(intArrayOf(1), "1,2,3".splitToInts(1))
        assertArrayEquals(intArrayOf(1), "1".splitToInts(-1))
    }

    @Test(expected = AssertionError::class)
    fun splitToIntsWrongLength() {
        assertArrayEquals(intArrayOf(1,2,3), "1,2,3".splitToInts(5))
    }

    @Test(expected = NumberFormatException::class)
    fun splitToIntsParsingError() {
        assertArrayEquals(intArrayOf(1,2,3), "1,b,3".splitToInts(3))
    }

    @Test(expected = NumberFormatException::class)
    fun splitToIntsParsingErrorDotValue1() {
        assertArrayEquals(intArrayOf(0), ".".splitToInts(-1))
    }

    @Test(expected = NumberFormatException::class)
    fun splitToIntsParsingErrorDotValue2() {
        assertArrayEquals(intArrayOf(0,0,0), ".".splitToInts(3))
    }

    @Test(expected = IOException::class)
    fun testCloseParser() {
        val format = BedFormat.DEFAULT
        withBedFile(CONTENT_RGB_EXAMPLE) { path ->
            val reader = path.bufferedReader()
            format.parse(reader) {
                // here may be some code
            }
            reader.read()
        }
    }

    @Test
    fun testWriteEmptyName() {
        val entries = listOf(
                ExtendedBedEntry("chr1", 1, 5, name="", strand = '+'),
                ExtendedBedEntry("chr1", 1, 5, name="", strand = '-'))

        withBedFile { trackPath ->
            val format = BedFormat()
            format.print(trackPath).use { bedPrinter ->
                entries.forEach { e -> bedPrinter.print(e) }
            }

            assertEquals(format, BedFormat.auto(trackPath))
            assertEquals(entries.map { it.copy(name = ".") },
                         format.parse(trackPath) {
                             it.map { e ->
                                 e.unpack(format)
                             }.toList()
                         })
        }
    }

    @Test
    fun testToBedEntry() {
        val chr = GenomeQuery("to1").get()[0]
        val loci = listOf(
                Location(1, 5, chr, Strand.PLUS),
                Location(1, 5, chr, Strand.MINUS))

        withBedFile { trackPath ->
            val bedFormat = BedFormat()
            bedFormat.print(trackPath).use { bedPrinter ->
                for (l in loci) {
                    bedPrinter.print(l.toBedEntry())
                }
            }

            assertEquals(bedFormat, BedFormat.auto(trackPath))
            assertEquals(loci, BedFormat.auto(trackPath).parseLocations(trackPath, "to1"))
        }
    }


    @Test
    fun fromFormatString() {
        BedFormat(3, 0, '\t').let { f ->
            assertEquals(f, BedFormat.fromString(f.toString()))
        }

        BedFormat(3, 0, ',').let { f ->
            assertEquals(f, BedFormat.fromString(f.toString()))
        }

        BedFormat(3, 0, ' ').let { f ->
            assertEquals(f, BedFormat.fromString(f.toString()))
        }

        BedFormat(4, 0).let { f ->
            assertEquals(f, BedFormat.fromString(f.toString()))
        }

        BedFormat(4, 2).let { f ->
            assertEquals(f, BedFormat.fromString(f.toString()))
        }

        BedFormat(4, null).let { f ->
            assertEquals(f, BedFormat.fromString(f.toString()))
        }
    }

    private fun withBedFile(contents: String = "", block: (Path) -> Unit) = withFile("bed", contents, block)

    private fun withFile(ext: String?, contents: String = "", block: (Path) -> Unit) {
        withTempFile("test", if (ext != null) ".$ext" else "") { path ->
            if (contents.isNotEmpty()) {
                path.write(contents)
            }

            block(path)
        }
    }
}