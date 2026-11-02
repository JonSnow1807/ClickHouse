#include <gtest/gtest.h>
#include <Common/CacheBase.h>
#include <Common/CurrentMetrics.h>
#include <Common/Exception.h>

#include <algorithm>
#include <atomic>
#include <future>
#include <limits>
#include <memory>
#include <set>
#include <thread>
#include <vector>

/// Use MarkCache* for tests (to avoid introducing one more metric)
namespace CurrentMetrics
{
    extern const Metric MarkCacheBytes;
    extern const Metric MarkCacheFiles;
}

namespace
{

struct ValueWeight
{
    size_t operator()(const size_t & x) const { return x; }
};

using Cache = DB::CacheBase<int, size_t, std::hash<int>, ValueWeight>;

constexpr size_t NO_LIMIT = 1'000'000'000;

Cache makeCache(const std::string & policy, size_t num_shards, size_t max_size_in_bytes = NO_LIMIT, size_t max_count = 0)
{
    return Cache(policy, CurrentMetrics::end(), CurrentMetrics::end(), max_size_in_bytes, max_count, /*size_ratio*/ 0.5, num_shards);
}

std::set<int> keysOf(const Cache & cache)
{
    std::set<int> keys;
    for (const auto & entry : cache.dump())
        keys.insert(entry.key);
    return keys;
}

}

TEST(CacheBaseShards, DefaultIsOneShard)
{
    Cache by_default_policy(CurrentMetrics::end(), CurrentMetrics::end(), NO_LIMIT);
    ASSERT_EQ(by_default_policy.numberOfShards(), 1u);

    Cache by_policy_name("SLRU", CurrentMetrics::end(), CurrentMetrics::end(), NO_LIMIT, 0, 0.5);
    ASSERT_EQ(by_policy_name.numberOfShards(), 1u);

    Cache by_policy(std::make_unique<DB::LRUCachePolicy<int, size_t, std::hash<int>, ValueWeight>>(
        CurrentMetrics::end(), CurrentMetrics::end(), NO_LIMIT, 0, nullptr));
    ASSERT_EQ(by_policy.numberOfShards(), 1u);
}

TEST(CacheBaseShards, ZeroShardsIsRejected)
{
    EXPECT_THROW(makeCache("SLRU", 0), DB::Exception);
    EXPECT_THROW(Cache(CurrentMetrics::end(), CurrentMetrics::end(), NO_LIMIT, 0, 0.5, 0), DB::Exception);
}

TEST(CacheBaseShards, EveryPolicyRoundTripsAcrossShards)
{
    for (const auto * policy : {"LRU", "SLRU", "SIEVE"})
    {
        auto cache = makeCache(policy, 4);
        ASSERT_EQ(cache.numberOfShards(), 4u) << policy;

        for (int key = 0; key < 1000; ++key)
            cache.set(key, std::make_shared<size_t>(key + 1));

        ASSERT_EQ(cache.count(), 1000u) << policy;
        ASSERT_EQ(cache.sizeInBytes(), 1000u * 1001u / 2) << policy;

        for (int key = 0; key < 1000; ++key)
        {
            ASSERT_TRUE(cache.contains(key)) << policy << " key " << key;
            auto value = cache.get(key);
            ASSERT_NE(value, nullptr) << policy << " key " << key;
            ASSERT_EQ(*value, static_cast<size_t>(key + 1)) << policy << " key " << key;

            auto with_key = cache.getWithKey(key);
            ASSERT_TRUE(with_key.has_value()) << policy;
            ASSERT_EQ(with_key->key, key) << policy;
        }

        ASSERT_EQ(cache.get(1000), nullptr) << policy;
        ASSERT_FALSE(cache.contains(1000)) << policy;
    }
}

TEST(CacheBaseShards, DefaultPolicyConstructorShards)
{
    Cache cache(CurrentMetrics::MarkCacheBytes, CurrentMetrics::MarkCacheFiles, NO_LIMIT, 0, 0.5, /*num_shards*/ 8);
    ASSERT_EQ(cache.numberOfShards(), 8u);

    const auto bytes_before = CurrentMetrics::get(CurrentMetrics::MarkCacheBytes);
    const auto files_before = CurrentMetrics::get(CurrentMetrics::MarkCacheFiles);

    for (int key = 0; key < 100; ++key)
        cache.set(key, std::make_shared<size_t>(2));

    /// The shards update the same metrics.
    ASSERT_EQ(CurrentMetrics::get(CurrentMetrics::MarkCacheBytes) - bytes_before, 200);
    ASSERT_EQ(CurrentMetrics::get(CurrentMetrics::MarkCacheFiles) - files_before, 100);

    cache.clear();
    ASSERT_EQ(CurrentMetrics::get(CurrentMetrics::MarkCacheBytes), bytes_before);
    ASSERT_EQ(CurrentMetrics::get(CurrentMetrics::MarkCacheFiles), files_before);
}

TEST(CacheBaseShards, SizeLimitIsSplitAndReportedAsConfigured)
{
    /// 10 bytes over 4 shards: shares of 3, 3, 2 and 2 bytes.
    auto cache = makeCache("SLRU", 4, /*max_size_in_bytes*/ 10);
    ASSERT_EQ(cache.maxSizeInBytes(), 10u);

    for (int key = 0; key < 100; ++key)
        cache.set(key, std::make_shared<size_t>(1));

    ASSERT_LE(cache.sizeInBytes(), 10u);
    ASSERT_LE(cache.count(), 10u);
    ASSERT_GT(cache.count(), 0u);

    /// An entry that does not fit into the share of any shard is never cached.
    cache.set(1000, std::make_shared<size_t>(4));
    ASSERT_FALSE(cache.contains(1000));

    /// Shrinking splits the new limit over the shards again.
    cache.setMaxSizeInBytes(4);
    ASSERT_EQ(cache.maxSizeInBytes(), 4u);
    ASSERT_LE(cache.sizeInBytes(), 4u);

    /// 7 bytes over 4 shards: 2, 2, 2, 1.
    cache.setMaxSizeInBytes(7);
    ASSERT_EQ(cache.maxSizeInBytes(), 7u);
}

TEST(CacheBaseShards, CountLimitIsSplitRoundedUp)
{
    /// 6 entries over 4 shards: a share of 2 each, so the cache reports 8 and holds at most 8.
    auto cache = makeCache("SLRU", 4, NO_LIMIT, /*max_count*/ 6);
    ASSERT_EQ(cache.maxCount(), 8u);

    for (int key = 0; key < 100; ++key)
        cache.set(key, std::make_shared<size_t>(1));

    ASSERT_LE(cache.count(), 8u);
    ASSERT_GT(cache.count(), 0u);

    /// A share rounded down to zero would mean no limit, so 3 entries over 4 shards give one per shard.
    cache.setMaxCount(3);
    ASSERT_EQ(cache.maxCount(), 4u);
    for (int key = 100; key < 200; ++key)
        cache.set(key, std::make_shared<size_t>(1));
    ASSERT_LE(cache.count(), 4u);

    /// A limit at the top of size_t is a legal server setting; its rounded-up shares must not wrap the reported
    /// limit around to the zero that means no limit.
    auto huge = makeCache("SLRU", 4, NO_LIMIT, std::numeric_limits<size_t>::max());
    ASSERT_EQ(huge.maxCount(), std::numeric_limits<size_t>::max());
    huge.set(1, std::make_shared<size_t>(1));
    ASSERT_TRUE(huge.contains(1));

    /// No limit stays no limit in every shard.
    auto unlimited = makeCache("LRU", 4, NO_LIMIT, Cache::NO_MAX_COUNT);
    ASSERT_EQ(unlimited.maxCount(), Cache::NO_MAX_COUNT);
    for (int key = 0; key < 1000; ++key)
        unlimited.set(key, std::make_shared<size_t>(1));
    ASSERT_EQ(unlimited.count(), 1000u);
}

TEST(CacheBaseShards, ClearEmptiesEveryShard)
{
    auto cache = makeCache("SLRU", 4);
    for (int key = 0; key < 100; ++key)
        cache.set(key, std::make_shared<size_t>(1));
    for (int key = 0; key < 50; ++key)
        cache.get(key);

    cache.clear();

    ASSERT_EQ(cache.count(), 0u);
    ASSERT_EQ(cache.sizeInBytes(), 0u);
    ASSERT_TRUE(cache.dump().empty());
    for (int key = 0; key < 100; ++key)
        ASSERT_EQ(cache.get(key), nullptr);

    /// `clear` also resets the statistics; the lookups above are the only ones counted after it.
    size_t hits = 0;
    size_t misses = 0;
    cache.getStats(hits, misses);
    ASSERT_EQ(hits, 0u);
    ASSERT_EQ(misses, 100u);
}

TEST(CacheBaseShards, RemoveByKeyAndByPredicate)
{
    auto cache = makeCache("SLRU", 4);
    for (int key = 0; key < 100; ++key)
        cache.set(key, std::make_shared<size_t>(1));

    cache.remove(7);
    ASSERT_FALSE(cache.contains(7));
    ASSERT_EQ(cache.count(), 99u);

    cache.remove([](const int & key, const Cache::MappedPtr &) { return key % 2 == 0; });
    ASSERT_EQ(cache.count(), 49u);
    for (int key = 0; key < 100; ++key)
        ASSERT_EQ(cache.contains(key), key % 2 == 1 && key != 7) << key;
}

TEST(CacheBaseShards, DumpCoversEveryShard)
{
    auto cache = makeCache("SIEVE", 4);
    std::set<int> expected;
    for (int key = 0; key < 100; ++key)
    {
        cache.set(key, std::make_shared<size_t>(1));
        expected.insert(key);
    }

    const auto dumped = cache.dump();
    ASSERT_EQ(dumped.size(), 100u);
    ASSERT_EQ(keysOf(cache), expected);
}

TEST(CacheBaseShards, StatsAreSummedOverShards)
{
    auto cache = makeCache("SLRU", 4);
    for (int key = 0; key < 50; ++key)
        cache.set(key, std::make_shared<size_t>(1));

    for (int key = 0; key < 100; ++key)
        cache.get(key);

    size_t hits = 0;
    size_t misses = 0;
    cache.getStats(hits, misses);
    ASSERT_EQ(hits, 50u);
    ASSERT_EQ(misses, 50u);
}

TEST(CacheBaseShards, ManyOperationsGroupKeysByShard)
{
    auto cache = makeCache("SLRU", 4);

    std::vector<int> keys;
    std::vector<Cache::MappedPtr> values;
    for (int key = 0; key < 100; ++key)
    {
        keys.push_back(key);
        values.push_back(std::make_shared<size_t>(key + 1));
    }

    cache.setMany(keys, values);
    ASSERT_EQ(cache.count(), 100u);

    const auto contained = cache.containsMany(keys);
    ASSERT_EQ(contained.size(), 100u);
    ASSERT_TRUE(std::all_of(contained.begin(), contained.end(), [](bool c) { return c; }));

    /// The results come back at the positions of their keys, whatever shard the keys live in.
    std::vector<int> lookups;
    for (int key = 99; key >= -50; key -= 3)
        lookups.push_back(key);

    const auto found = cache.getMany(lookups);
    ASSERT_EQ(found.size(), lookups.size());
    for (size_t i = 0; i < lookups.size(); ++i)
    {
        if (lookups[i] >= 0)
        {
            ASSERT_NE(found[i], nullptr) << lookups[i];
            ASSERT_EQ(*found[i], static_cast<size_t>(lookups[i] + 1)) << lookups[i];
        }
        else
            ASSERT_EQ(found[i], nullptr) << lookups[i];
    }

    size_t hits = 0;
    size_t misses = 0;
    cache.getStats(hits, misses);
    const size_t expected_hits = static_cast<size_t>(std::count_if(lookups.begin(), lookups.end(), [](int key) { return key >= 0; }));
    ASSERT_EQ(hits, expected_hits);
    ASSERT_EQ(misses, lookups.size() - expected_hits);
}

TEST(CacheBaseShards, ConditionalOperationsUseTheShardOfTheKey)
{
    auto cache = makeCache("SLRU", 4);
    auto first = std::make_shared<size_t>(1);
    auto second = std::make_shared<size_t>(2);

    for (int key = 0; key < 100; ++key)
        ASSERT_TRUE(cache.setAndContains(key, first, [] {}));

    for (int key = 0; key < 100; ++key)
    {
        ASSERT_FALSE(cache.replaceIfMatches(key, second, second)) << key;
        ASSERT_TRUE(cache.replaceIfMatches(key, first, second)) << key;
        ASSERT_EQ(cache.get(key), second) << key;
    }

    for (int key = 0; key < 100; ++key)
    {
        ASSERT_FALSE(cache.updateIfMatches(key, [](const Cache::MappedPtr &) { return false; })) << key;
        ASSERT_TRUE(cache.updateIfMatches(key, [](const Cache::MappedPtr & value) { *value = 3; return true; })) << key;
    }
    ASSERT_EQ(cache.sizeInBytes(), 300u);

    for (int key = 0; key < 100; ++key)
    {
        ASSERT_FALSE(cache.removeIfMatches(key, [](const Cache::MappedPtr &) { return false; })) << key;
        ASSERT_TRUE(cache.removeIfMatches(key, [](const Cache::MappedPtr & value) { return *value == 3; })) << key;
    }
    ASSERT_EQ(cache.count(), 0u);
}

TEST(CacheBaseShards, GetOrSetLoadsEveryKeyOnceUnderConcurrency)
{
    constexpr size_t num_threads = 16;
    constexpr int num_keys = 1000;

    auto cache = makeCache("SLRU", 8);

    std::vector<std::atomic<size_t>> loads(num_keys);
    std::atomic<size_t> wrong_values{0};

    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&, t]
        {
            /// The threads walk the keys from different starting points, so that they meet on every key.
            for (int i = 0; i < num_keys; ++i)
            {
                const int key = static_cast<int>((i + t * 61) % num_keys);
                auto [value, loaded] = cache.getOrSet(key, [&]
                {
                    ++loads[key];
                    return std::make_shared<size_t>(key + 1);
                });
                if (!value || *value != static_cast<size_t>(key + 1))
                    ++wrong_values;
            }
        });
    }
    for (auto & thread : threads)
        thread.join();

    ASSERT_EQ(wrong_values.load(), 0u);
    for (int key = 0; key < num_keys; ++key)
        ASSERT_EQ(loads[key].load(), 1u) << key;
    ASSERT_EQ(cache.count(), static_cast<size_t>(num_keys));

    /// One miss per key - for the thread that loaded it - and a hit for every other lookup, including the lookups
    /// that waited for a load in progress.
    size_t hits = 0;
    size_t misses = 0;
    cache.getStats(hits, misses);
    ASSERT_EQ(misses, static_cast<size_t>(num_keys));
    ASSERT_EQ(hits, num_threads * num_keys - num_keys);
}

TEST(CacheBaseShards, ConcurrentClearKeepsGetOrSetConsistent)
{
    /// A `clear` while loads are in flight: the loaded value is still returned to the loader, and nothing crashes or
    /// leaks an insert token. The values the cache holds afterwards are correct.
    constexpr size_t num_threads = 8;
    constexpr int num_keys = 500;

    auto cache = makeCache("SLRU", 4);
    std::atomic<bool> stop{false};
    std::atomic<size_t> wrong_values{0};

    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&]
        {
            for (int round = 0; round < 20; ++round)
                for (int key = 0; key < num_keys; ++key)
                {
                    auto [value, loaded] = cache.getOrSet(key, [&] { return std::make_shared<size_t>(key + 1); });
                    if (!value || *value != static_cast<size_t>(key + 1))
                        ++wrong_values;
                }
        });
    }
    std::thread clearer([&]
    {
        while (!stop)
            cache.clear();
    });

    for (auto & thread : threads)
        thread.join();
    stop = true;
    clearer.join();

    ASSERT_EQ(wrong_values.load(), 0u);
    for (const auto & entry : cache.dump())
        ASSERT_EQ(*entry.mapped, static_cast<size_t>(entry.key + 1));
}

TEST(CacheBaseShards, LoaderFinishingAfterClearDoesNotDropTheNextLoadersToken)
{
    /// Loader A is in flight for a key when the cache is cleared, and loader B starts for the same key while A is still
    /// loading. A's value has to stay out of the cache (its token was cleared), and A's cleanup has to leave B's token
    /// alone, so that B's value is cached when B finishes. With one shard this is the protocol of the unsharded cache.
    auto cache = makeCache("SLRU", 1);

    std::promise<void> a_started;
    std::promise<void> a_may_finish;
    std::promise<void> b_started;
    std::promise<void> b_may_finish;
    auto a_started_future = a_started.get_future();
    auto a_may_finish_future = a_may_finish.get_future();
    auto b_started_future = b_started.get_future();
    auto b_may_finish_future = b_may_finish.get_future();

    std::thread loader_a([&]
    {
        cache.getOrSet(1, [&]
        {
            a_started.set_value();
            a_may_finish_future.wait();
            return std::make_shared<size_t>(10);
        });
    });
    a_started_future.wait();

    cache.clear();

    std::thread loader_b([&]
    {
        cache.getOrSet(1, [&]
        {
            b_started.set_value();
            b_may_finish_future.wait();
            return std::make_shared<size_t>(20);
        });
    });
    b_started_future.wait();

    a_may_finish.set_value();
    loader_a.join();

    b_may_finish.set_value();
    loader_b.join();

    auto value = cache.get(1);
    ASSERT_NE(value, nullptr);
    ASSERT_EQ(*value, 20u);
    ASSERT_EQ(cache.count(), 1u);
}

TEST(CacheBaseShards, DefaultArgumentMeansOneShard)
{
    /// The same sequence of operations on a cache constructed with and without the parameter gives the same contents
    /// and statistics: the default is one shard. (The behaviour of the one-shard cache itself is what the LRUCache,
    /// SLRUCache and SIEVECache suites check.)
    auto run = [](Cache & cache)
    {
        for (int key = 0; key < 20; ++key)
            cache.set(key, std::make_shared<size_t>(key % 4 + 1));
        for (int key = 0; key < 20; key += 2)
            cache.get(key);
        for (int key = 20; key < 40; ++key)
            cache.getOrSet(key, [&] { return std::make_shared<size_t>(2); });
        cache.remove(3);
        cache.setMaxSizeInBytes(10);
    };

    Cache implicit("SLRU", CurrentMetrics::end(), CurrentMetrics::end(), 30, 0, 0.5);
    Cache explicit_one("SLRU", CurrentMetrics::end(), CurrentMetrics::end(), 30, 0, 0.5, 1);
    run(implicit);
    run(explicit_one);

    ASSERT_EQ(keysOf(implicit), keysOf(explicit_one));
    ASSERT_EQ(implicit.sizeInBytes(), explicit_one.sizeInBytes());
    ASSERT_EQ(implicit.maxSizeInBytes(), explicit_one.maxSizeInBytes());

    size_t hits_implicit = 0;
    size_t misses_implicit = 0;
    size_t hits_explicit = 0;
    size_t misses_explicit = 0;
    implicit.getStats(hits_implicit, misses_implicit);
    explicit_one.getStats(hits_explicit, misses_explicit);
    ASSERT_EQ(hits_implicit, hits_explicit);
    ASSERT_EQ(misses_implicit, misses_explicit);
}
