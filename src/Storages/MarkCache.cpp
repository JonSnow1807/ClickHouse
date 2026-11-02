#include <Common/SipHash.h>
#include <Common/getNumberOfCPUCoresToUse.h>
#include <Storages/MarkCache.h>

#include <algorithm>

namespace DB
{
template class CacheBase<UInt128, MarksInCompressedFile, UInt128TrivialHash, MarksWeightFunction>;

MarkCache::MarkCache(const String & cache_policy,
    CurrentMetrics::Metric size_in_bytes_metric,
    CurrentMetrics::Metric count_metric,
    size_t max_size_in_bytes,
    double size_ratio,
    size_t num_shards)
    : Base(cache_policy, size_in_bytes_metric, count_metric, max_size_in_bytes, 0, size_ratio, num_shards)
{
}

size_t MarkCache::chooseNumberOfShards(size_t configured_num_shards, size_t max_size_in_bytes)
{
    if (configured_num_shards != 0)
        return configured_num_shards;

    const size_t shards_by_size = max_size_in_bytes / AUTO_BYTES_PER_SHARD;
    const size_t shards_by_cores = getNumberOfCPUCoresToUse();
    return std::clamp(std::min(shards_by_size, shards_by_cores), 1uz, MAX_AUTO_SHARDS);
}

UInt128 MarkCache::hash(const String& path_to_file)
{
    SipHash hash;
    hash.update(path_to_file.size());
    hash.update(path_to_file.data(), path_to_file.size());
    return hash.get128();
}


}
