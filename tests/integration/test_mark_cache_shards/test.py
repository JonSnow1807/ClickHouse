import pytest

from helpers.cluster import ClickHouseCluster

cluster = ClickHouseCluster(__file__)
node = cluster.add_instance("node", main_configs=["configs/mark_cache_shards.xml"])

TABLE = "mark_cache_shards_test"
NUM_PARTS = 16
# Every lookup reads one wide part, and at least the three selected columns of it have a marks file each.
MIN_MARKS_FILES_PER_LOOKUP = 3


@pytest.fixture(scope="module")
def started_cluster():
    try:
        cluster.start()
        yield cluster
    finally:
        cluster.shutdown()


def metric(name):
    return int(node.query(f"SELECT sum(value) FROM system.metrics WHERE metric = '{name}'").strip())


def event(name):
    # `system.events` leaves out the events that have not happened yet, hence the sum.
    return int(node.query(f"SELECT sum(value) FROM system.events WHERE event = '{name}'").strip())


def point_lookups():
    # The parts hold disjoint ranges of `id`, so the primary key sends each lookup to exactly one part.
    for i in range(NUM_PARTS):
        node.query(f"SELECT a, b, c FROM {TABLE} WHERE id = {i * 1000 + 7} FORMAT Null")


def test_setting_is_applied(started_cluster):
    assert (
        node.query("SELECT value FROM system.server_settings WHERE name = 'mark_cache_shards'").strip()
        == "8"
    )
    # The server logs the number of shards the cache was built with.
    assert node.contains_in_log("Mark cache is split into 8 shard(s)")


def test_hits_misses_and_drop_with_shards(started_cluster):
    node.query(f"DROP TABLE IF EXISTS {TABLE}")
    node.query(
        f"""
        CREATE TABLE {TABLE} (id UInt64, a String, b UInt64, c Float64)
        ENGINE = MergeTree ORDER BY id
        SETTINGS index_granularity = 64
        """
    )
    node.query(f"SYSTEM STOP MERGES {TABLE}")
    for i in range(NUM_PARTS):
        node.query(
            f"INSERT INTO {TABLE} SELECT number + {i * 1000}, toString(number), number * 2, number / 3 FROM numbers(1000)"
        )
    assert (
        int(node.query(f"SELECT count() FROM system.parts WHERE table = '{TABLE}' AND active"))
        == NUM_PARTS
    )

    node.query("SYSTEM DROP MARK CACHE")
    assert metric("MarkCacheFiles") == 0

    # The cache is empty, so the first round loads the marks of every part it reads.
    misses_before = event("MarkCacheMisses")
    point_lookups()
    assert event("MarkCacheMisses") - misses_before >= MIN_MARKS_FILES_PER_LOOKUP * NUM_PARTS
    files_after_first_round = metric("MarkCacheFiles")
    assert files_after_first_round >= MIN_MARKS_FILES_PER_LOOKUP * NUM_PARTS

    # The second round finds them, in whichever shard the hash of their path put them.
    hits_before = event("MarkCacheHits")
    misses_before = event("MarkCacheMisses")
    point_lookups()
    assert event("MarkCacheHits") - hits_before >= MIN_MARKS_FILES_PER_LOOKUP * NUM_PARTS
    assert event("MarkCacheMisses") == misses_before
    assert metric("MarkCacheFiles") == files_after_first_round

    # `SYSTEM DROP MARK CACHE` clears every shard.
    node.query("SYSTEM DROP MARK CACHE")
    assert metric("MarkCacheFiles") == 0
    assert metric("MarkCacheBytes") == 0

    # And the cache fills again after that.
    point_lookups()
    assert metric("MarkCacheFiles") >= MIN_MARKS_FILES_PER_LOOKUP * NUM_PARTS

    node.query(f"DROP TABLE {TABLE}")
