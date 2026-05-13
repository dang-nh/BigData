"""Unit tests for ER feature UDFs."""

import pytest


def test_jaro_winkler_same():
    from pyspark.sql import SparkSession
    from src.er.features import jaro_winkler

    spark = SparkSession.builder.master("local").appName("test").getOrCreate()
    df = spark.createDataFrame([("ASPIRIN", "ASPIRIN")], ["a", "b"])
    result = df.select(jaro_winkler("a", "b").alias("jw")).collect()[0]["jw"]
    assert result == pytest.approx(1.0, abs=0.01)
    spark.stop()


def test_jaro_winkler_different():
    from pyspark.sql import SparkSession
    from src.er.features import jaro_winkler

    spark = SparkSession.builder.master("local").appName("test").getOrCreate()
    df = spark.createDataFrame([("ASPIRIN", "WARFARIN")], ["a", "b"])
    result = df.select(jaro_winkler("a", "b").alias("jw")).collect()[0]["jw"]
    assert result < 0.8
    spark.stop()


def test_token_overlap_partial():
    from pyspark.sql import SparkSession
    from src.er.features import token_overlap

    spark = SparkSession.builder.master("local").appName("test").getOrCreate()
    df = spark.createDataFrame([("HEART FAILURE ACUTE", "HEART FAILURE CHRONIC")], ["a", "b"])
    result = df.select(token_overlap("a", "b").alias("tok")).collect()[0]["tok"]
    assert 0.4 < result < 0.8
    spark.stop()


def test_icd_chapter_udf():
    from pyspark.sql import SparkSession
    from src.er.blocking import icd_chapter

    spark = SparkSession.builder.master("local").appName("test").getOrCreate()
    df = spark.createDataFrame([("41401",), ("0010",)], ["code"])
    results = df.select(icd_chapter("code").alias("ch")).collect()
    chapters = [r["ch"] for r in results]
    assert "CIRCULATORY" in chapters
    assert "INFECTIOUS" in chapters
    spark.stop()
