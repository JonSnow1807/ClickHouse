#pragma once

#include <Common/Exception.h>
#include <Common/ICachePolicy.h>
#include <Common/LRUCachePolicy.h>
#include <Common/SLRUCachePolicy.h>
#include <Common/SIEVECachePolicy.h>
#include <Common/CurrentMetrics.h>
#include <Common/CacheLine.h>
#include <Common/HashTable/Hash.h>

#include <base/UUID.h>
#include <base/defines.h>

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>


namespace DB
{
namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
}

/// Thread-safe cache that evicts entries using special cache policy
/// (default policy evicts entries which are not used for a long time).
/// WeightFunction is a functor that takes Mapped as a parameter and returns "weight" (approximate size)
/// of that value.
/// Cache starts to evict entries when their total weight exceeds max_size_in_bytes.
/// Value weight should not change after insertion.
///
/// The cache can be striped over several shards. A shard is a policy instance with a lock of its own, and a key
/// always lives in the shard that its hash selects, so threads that work on keys of different shards do not wait
/// for each other. That is what a cache with many small entries and a high hit rate needs on a machine with many
/// cores: with one lock, every hit of every query thread queues on it. The price is that the policy applies per
/// shard - each shard evicts by its own order and, for SLRU, keeps its own protected segment - and so do the limits:
/// every shard gets an equal share of `max_size_in_bytes` and of `max_count`, an entry larger than the share of its
/// shard is never cached, and the totals the cache reports are sums over the shards that are not taken under one lock.
/// `onEntryRemoval` runs under the lock of the shard that evicts, so with several shards it can run concurrently for
/// different shards; a subclass that asks for more than one shard has to be ready for that.
/// With one shard, the default, the cache behaves exactly as it did without shards.
template <typename TKey, typename TMapped, typename HashFunction = std::hash<TKey>, typename WeightFunction = EqualWeightFunction<TMapped>>
class CacheBase
{
private:
    using CachePolicy = ICachePolicy<TKey, TMapped, HashFunction, WeightFunction>;

    struct Shard;

public:
    using Key = typename CachePolicy::Key;
    using Mapped = typename CachePolicy::Mapped;
    using MappedPtr = typename CachePolicy::MappedPtr;
    using KeyMapped = typename CachePolicy::KeyMapped;

    static constexpr auto NO_MAX_COUNT = 0uz;
    static constexpr auto DEFAULT_SIZE_RATIO = 0.5l;
    static constexpr auto DEFAULT_CACHE_POLICY = "SLRU";
    static constexpr auto DEFAULT_NUM_SHARDS = 1uz;

    /// Use this ctor if you only care about the cache size but not internals like the cache policy.
    explicit CacheBase(
        CurrentMetrics::Metric size_in_bytes_metric,
        CurrentMetrics::Metric count_metric,
        size_t max_size_in_bytes,
        size_t max_count = NO_MAX_COUNT,
        double size_ratio = DEFAULT_SIZE_RATIO,
        size_t num_shards = DEFAULT_NUM_SHARDS)
    {
        /// Construct the default policy directly instead of delegating to the ctor below, so that caches which never
        /// select a policy by name do not instantiate the code of the other policies.
        using SLRUPolicy = SLRUCachePolicy<TKey, TMapped, HashFunction, WeightFunction>;
        createShards(num_shards, max_size_in_bytes, max_count, [&](size_t shard_max_size_in_bytes, size_t shard_max_count)
        {
            return std::make_unique<SLRUPolicy>(size_in_bytes_metric, count_metric, shard_max_size_in_bytes, shard_max_count, size_ratio, makeOnRemoveEntryFunction());
        });
    }

    /// Use this ctor if the user should be able to configure the cache policy and cache sizes via settings. Supports only general-purpose policies LRU, SLRU and SIEVE.
    explicit CacheBase(
        std::string_view cache_policy_name,
        CurrentMetrics::Metric size_in_bytes_metric,
        CurrentMetrics::Metric count_metric,
        size_t max_size_in_bytes,
        size_t max_count,
        double size_ratio,
        size_t num_shards = DEFAULT_NUM_SHARDS)
    {
        auto on_remove_entry_function = makeOnRemoveEntryFunction();

        if (cache_policy_name.empty())
        {
            cache_policy_name = DEFAULT_CACHE_POLICY;
        }

        if (cache_policy_name == "LRU")
        {
            using LRUPolicy = LRUCachePolicy<TKey, TMapped, HashFunction, WeightFunction>;
            createShards(num_shards, max_size_in_bytes, max_count, [&](size_t shard_max_size_in_bytes, size_t shard_max_count)
            {
                return std::make_unique<LRUPolicy>(size_in_bytes_metric, count_metric, shard_max_size_in_bytes, shard_max_count, on_remove_entry_function);
            });
        }
        else if (cache_policy_name == "SLRU")
        {
            using SLRUPolicy = SLRUCachePolicy<TKey, TMapped, HashFunction, WeightFunction>;
            createShards(num_shards, max_size_in_bytes, max_count, [&](size_t shard_max_size_in_bytes, size_t shard_max_count)
            {
                return std::make_unique<SLRUPolicy>(size_in_bytes_metric, count_metric, shard_max_size_in_bytes, shard_max_count, size_ratio, on_remove_entry_function);
            });
        }
        else if (cache_policy_name == "SIEVE")
        {
            using SIEVEPolicy = SIEVECachePolicy<TKey, TMapped, HashFunction, WeightFunction>;
            createShards(num_shards, max_size_in_bytes, max_count, [&](size_t shard_max_size_in_bytes, size_t shard_max_count)
            {
                return std::make_unique<SIEVEPolicy>(size_in_bytes_metric, count_metric, shard_max_size_in_bytes, shard_max_count, on_remove_entry_function);
            });
        }
        else
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Unknown cache policy name: {}", cache_policy_name);
    }

    /// Use this ctor to provide an arbitrary cache policy. Such a cache has one shard.
    explicit CacheBase(std::unique_ptr<ICachePolicy<TKey, TMapped, HashFunction, WeightFunction>> cache_policy_)
    {
        shards.push_back(std::make_unique<Shard>(std::move(cache_policy_)));
    }

    size_t numberOfShards() const { return shards.size(); }

    MappedPtr get(const Key & key)
    {
        Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        auto res = shard.cache_policy->get(key);
        if (res)
            ++shard.hits;
        else
            ++shard.misses;
        return res;
    }

    std::optional<KeyMapped> getWithKey(const Key & key)
    {
        Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        auto res = shard.cache_policy->getWithKey(key);
        if (res.has_value())
            ++shard.hits;
        else
            ++shard.misses;
        return res;
    }

    /// Look up several keys. Each shard is locked once for all of its keys.
    std::vector<MappedPtr> getMany(const std::vector<Key> & keys)
    {
        std::vector<MappedPtr> values;
        values.resize(keys.size());

        if (shards.size() == 1)
        {
            Shard & shard = *shards.front();
            std::lock_guard lock(shard.mutex);

            for (size_t i = 0; i < keys.size(); ++i)
            {
                auto res = shard.cache_policy->get(keys[i]);
                if (res)
                    ++shard.hits;
                else
                    ++shard.misses;

                values[i] = std::move(res);
            }

            return values;
        }

        const auto positions_by_shard = positionsByShard(keys);
        for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index)
        {
            const auto & positions = positions_by_shard[shard_index];
            if (positions.empty())
                continue;

            Shard & shard = *shards[shard_index];
            std::lock_guard lock(shard.mutex);

            for (size_t i : positions)
            {
                auto res = shard.cache_policy->get(keys[i]);
                if (res)
                    ++shard.hits;
                else
                    ++shard.misses;

                values[i] = std::move(res);
            }
        }

        return values;
    }

    /// Insert several entries. Each shard is locked once for all of its keys.
    void setMany(const std::vector<Key> & keys, const std::vector<MappedPtr> & mapped)
    {
        chassert(keys.size() == mapped.size());

        if (shards.size() == 1)
        {
            Shard & shard = *shards.front();
            std::lock_guard lock(shard.mutex);
            for (size_t i = 0; i < keys.size(); ++i)
                shard.cache_policy->set(keys[i], mapped[i]);
            return;
        }

        const auto positions_by_shard = positionsByShard(keys);
        for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index)
        {
            const auto & positions = positions_by_shard[shard_index];
            if (positions.empty())
                continue;

            Shard & shard = *shards[shard_index];
            std::lock_guard lock(shard.mutex);
            for (size_t i : positions)
                shard.cache_policy->set(keys[i], mapped[i]);
        }
    }

    /// Which of the keys are in the cache. Each shard is locked once for all of its keys. Does not touch the priorities.
    std::vector<bool> containsMany(const std::vector<Key> & keys) const
    {
        std::vector<bool> result(keys.size());

        if (shards.size() == 1)
        {
            const Shard & shard = *shards.front();
            std::lock_guard lock(shard.mutex);
            for (size_t i = 0; i < keys.size(); ++i)
                result[i] = shard.cache_policy->contains(keys[i]);
            return result;
        }

        const auto positions_by_shard = positionsByShard(keys);
        for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index)
        {
            const auto & positions = positions_by_shard[shard_index];
            if (positions.empty())
                continue;

            const Shard & shard = *shards[shard_index];
            std::lock_guard lock(shard.mutex);
            for (size_t i : positions)
                result[i] = shard.cache_policy->contains(keys[i]);
        }

        return result;
    }

    bool contains(const Key & key) const
    {
        const Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        return shard.cache_policy->contains(key);
    }

    void set(const Key & key, const MappedPtr & mapped)
    {
        Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        shard.cache_policy->set(key, mapped);
    }

    /// `set`, and whether `mapped` is in the cache right after it: the policy can refuse it or evict it at once.
    /// If it is, calls `on_resident` under the same lock, so nothing else can replace it before the call returns.
    template <typename OnResident>
    bool setAndContains(const Key & key, const MappedPtr & mapped, OnResident && on_resident)
    {
        Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        shard.cache_policy->set(key, mapped);
        if (!shard.cache_policy->contains(key))
            return false;
        on_resident();
        return true;
    }

    /// If the value for the key is in the cache, returns it. If it is not, calls load_func() to
    /// produce it, saves the result in the cache and returns it.
    /// Only one of several concurrent threads calling getOrSet() will call load_func(),
    /// others will wait for that call to complete and will use its result (this helps prevent cache stampede).
    /// Exceptions occurring in load_func will be propagated to the caller. Another thread from the
    /// set of concurrent threads will then try to call its load_func etc.
    ///
    /// Returns std::pair of the cached value and a bool indicating whether the value was produced during this call.
    template <typename LoadFunc>
    std::pair<MappedPtr, bool> getOrSet(const Key & key, LoadFunc && load_func)
    {
        Shard & shard = shardOf(key);

        InsertTokenHolder token_holder;
        {
            std::lock_guard cache_lock(shard.mutex);
            auto val = shard.cache_policy->get(key);
            if (val)
            {
                ++shard.hits;
                return std::make_pair(val, false);
            }

            auto & token = shard.insert_tokens[key];
            if (!token)
                token = std::make_shared<InsertToken>(shard);

            token_holder.acquire(&key, token, cache_lock);
        }

        InsertToken * token = token_holder.token.get();

        std::lock_guard token_lock(token->mutex);

        token_holder.cleaned_up = token->cleaned_up;

        if (token->value)
        {
            /// Another thread already produced the value while we waited for token->mutex.
            ++shard.hits;
            return std::make_pair(token->value, false);
        }

        ++shard.misses;
        token->value = load_func();

        std::lock_guard cache_lock(shard.mutex);

        /// Insert the new value only if the token is still in present in insert_tokens.
        /// (The token may be absent because of a concurrent clear() call).
        bool result = false;
        auto token_it = shard.insert_tokens.find(key);
        if (token_it != shard.insert_tokens.end() && token_it->second.get() == token)
        {
            shard.cache_policy->set(key, token->value);
            result = true;
        }

        if (!token->cleaned_up)
            token_holder.cleanup(token_lock, cache_lock);

        return std::make_pair(token->value, result);
    }

    /// The hits and misses of all shards. The shards are read one after another, so with several shards the
    /// numbers can come from slightly different moments.
    void getStats(size_t & out_hits, size_t & out_misses) const
    {
        out_hits = 0;
        out_misses = 0;
        for (const auto & shard_ptr : shards)
        {
            const Shard & shard = *shard_ptr;
            std::lock_guard lock(shard.mutex);
            out_hits += shard.hits;
            out_misses += shard.misses;
        }
    }

    /// The entries of all shards, shard by shard.
    std::vector<KeyMapped> dump() const
    {
        std::vector<KeyMapped> res;
        for (const auto & shard_ptr : shards)
        {
            const Shard & shard = *shard_ptr;
            std::lock_guard lock(shard.mutex);
            auto shard_entries = shard.cache_policy->dump();
            if (res.empty())
                res = std::move(shard_entries);
            else
                res.insert(res.end(), std::make_move_iterator(shard_entries.begin()), std::make_move_iterator(shard_entries.end()));
        }
        return res;
    }

    /// Clears the shards one after another: a value loaded for one shard while another one is being cleared is kept.
    void clear()
    {
        for (auto & shard_ptr : shards)
        {
            Shard & shard = *shard_ptr;
            std::lock_guard lock(shard.mutex);
            shard.insert_tokens.clear();
            shard.hits = 0;
            shard.misses = 0;
            shard.cache_policy->clear();
        }
    }

    void remove(const Key & key)
    {
        Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        shard.cache_policy->remove(key);
    }

    void remove(std::function<bool(const Key&, const MappedPtr &)> predicate)
    {
        forEachPolicy([&](CachePolicy & policy) { policy.remove(predicate); });
    }

    /// O(1) "remove iff" by key: looks up the resident under `key` and
    /// removes it only when `predicate(resident)` returns true. Returns
    /// whether the entry was removed. Atomic with respect to concurrent
    /// set/get/remove (single mutex acquisition for lookup + remove).
    ///
    /// Note: the predicate is invoked with a `MappedPtr` copy of the stored
    /// shared_ptr (cache_policy->get also bumps LRU position as a side
    /// effect), so the predicate sees `use_count() == stored_refs + 1`.
    /// Callers reasoning about use_count should account for the extra ref.
    bool removeIfMatches(const Key & key, std::function<bool(const MappedPtr &)> predicate)
    {
        Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        auto value = shard.cache_policy->get(key);
        if (!value || !predicate(value))
            return false;
        shard.cache_policy->remove(key);
        return true;
    }

    /// "Compare and set" by key: replaces the resident under `key` with `mapped` only when the
    /// resident is `expected` (the same object). Returns whether it was replaced. Atomic like
    /// removeIfMatches, and like it bumps the LRU position of the resident.
    bool replaceIfMatches(const Key & key, const MappedPtr & expected, const MappedPtr & mapped)
    {
        Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        auto value = shard.cache_policy->get(key);
        if (!value || value != expected)
            return false;
        shard.cache_policy->set(key, mapped);
        return true;
    }

    /// Like `removeIfMatches`, but if `update(resident)` returns true, sets the resident again to recompute its weight.
    /// `update` runs under the cache lock, so no eviction sees a changed resident with its old weight.
    bool updateIfMatches(const Key & key, std::function<bool(const MappedPtr &)> update)
    {
        Shard & shard = shardOf(key);
        std::lock_guard lock(shard.mutex);
        auto value = shard.cache_policy->get(key);
        if (!value || !update(value))
            return false;
        shard.cache_policy->set(key, value);
        return true;
    }

    /// The sizes, counts and limits below are sums over the shards, each read under the lock of its shard.

    size_t sizeInBytes() const
    {
        return sumOverPolicies([](const CachePolicy & policy) { return policy.sizeInBytes(); });
    }

    size_t count() const
    {
        return sumOverPolicies([](const CachePolicy & policy) { return policy.count(); });
    }

    size_t maxSizeInBytes() const
    {
        return sumOverPolicies([](const CachePolicy & policy) { return policy.maxSizeInBytes(); });
    }

    /// With several shards, a limit that does not divide by their number is rounded up to a multiple of it: every
    /// shard gets the same share, and a share rounded down to zero would mean no limit at all. A sum of shares that
    /// does not fit into size_t is reported as the largest size_t, never as the zero that means no limit.
    size_t maxCount() const
    {
        size_t res = 0;
        for (const auto & shard_ptr : shards)
        {
            const Shard & shard = *shard_ptr;
            std::lock_guard lock(shard.mutex);
            const size_t shard_max_count = shard.cache_policy->maxCount();
            res = shard_max_count > std::numeric_limits<size_t>::max() - res ? std::numeric_limits<size_t>::max() : res + shard_max_count;
        }
        return res;
    }

    void setMaxCount(size_t max_count)
    {
        std::lock_guard limits_lock(limits_mutex);
        const size_t shard_max_count = shardMaxCount(max_count, shards.size());
        forEachPolicy([&](CachePolicy & policy) { policy.setMaxCount(shard_max_count); });
    }

    void setMaxSizeInBytes(size_t max_size_in_bytes)
    {
        std::lock_guard limits_lock(limits_mutex);
        for (size_t shard_index = 0; shard_index < shards.size(); ++shard_index)
        {
            Shard & shard = *shards[shard_index];
            std::lock_guard lock(shard.mutex);
            shard.cache_policy->setMaxSizeInBytes(shardMaxSizeInBytes(max_size_in_bytes, shards.size(), shard_index));
        }
    }

    void setQuotaForUser(const UUID & user_id, size_t max_size_in_bytes, size_t max_entries)
    {
        forEachPolicy([&](CachePolicy & policy) { policy.setQuotaForUser(user_id, max_size_in_bytes, max_entries); });
    }

    /// For unit tests, and only for a cache with one shard.
    /// Do not use it anywhere else as this is not thread-safe.
    const CachePolicy & getCachePolicy()
    {
        chassert(shards.size() == 1);
        Shard & shard = *shards.front();
        std::lock_guard lock(shard.mutex);
        return *shard.cache_policy;
    }

    virtual ~CacheBase() = default;

private:
    struct InsertToken;
    using InsertTokenById = std::unordered_map<Key, std::shared_ptr<InsertToken>, HashFunction>;

    /// A stripe of the cache: a policy instance, the lock that guards it, the pending insertions into it and its
    /// statistics. The shards are independent of each other, which is the point: see the class comment.
    /// Each shard takes its own cache line(s), so that the threads of one shard do not slow down those of the next.
    struct alignas(CH_CACHE_LINE_SIZE) Shard
    {
        explicit Shard(std::unique_ptr<CachePolicy> cache_policy_) : cache_policy(std::move(cache_policy_)) {}

        mutable std::mutex mutex;
        std::unique_ptr<CachePolicy> cache_policy TSA_GUARDED_BY(mutex);
        InsertTokenById insert_tokens TSA_GUARDED_BY(mutex);
        std::atomic<size_t> hits{0};
        std::atomic<size_t> misses{0};
    };

    std::vector<std::unique_ptr<Shard>> shards;

    /// Serializes the setters that give every shard its share of a new limit, so that two concurrent calls cannot leave
    /// the shards with shares of different limits. Taken before the locks of the shards, never while one of them is held.
    std::mutex limits_mutex;

    /// Creates `num_shards` shards whose policies `make_policy(max_size_in_bytes, max_count)` builds, each with its share of the limits.
    template <typename MakePolicy>
    void createShards(size_t num_shards, size_t max_size_in_bytes, size_t max_count, MakePolicy && make_policy)
    {
        if (num_shards == 0)
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "The number of cache shards must be at least 1");

        shards.reserve(num_shards);
        for (size_t shard_index = 0; shard_index < num_shards; ++shard_index)
            shards.push_back(std::make_unique<Shard>(
                make_policy(shardMaxSizeInBytes(max_size_in_bytes, num_shards, shard_index), shardMaxCount(max_count, num_shards))));
    }

    /// The share of `max_size_in_bytes` of shard `shard_index` of `num_shards`: the shares differ by at most one byte and add up
    /// to `max_size_in_bytes` exactly, so the limit the cache reports is the configured one.
    static size_t shardMaxSizeInBytes(size_t max_size_in_bytes, size_t num_shards, size_t shard_index)
    {
        return max_size_in_bytes / num_shards + (shard_index < max_size_in_bytes % num_shards ? 1 : 0);
    }

    /// The share of `max_count` of each of `num_shards` shards, rounded up: a share of zero would mean no limit (`NO_MAX_COUNT`).
    static size_t shardMaxCount(size_t max_count, size_t num_shards)
    {
        if (max_count == NO_MAX_COUNT)
            return NO_MAX_COUNT;
        return max_count / num_shards + (max_count % num_shards != 0 ? 1 : 0);
    }

    size_t shardIndex(const Key & key) const
    {
        if (shards.size() == 1)
            return 0;
        /// The policies index their hash tables by `HashFunction` of the key. Mixing its bits (the finalizer of MurmurHash)
        /// before taking the remainder keeps the shard of a key independent of the bits those tables use, also when
        /// `HashFunction` is trivial, e.g. the identity of an integer key.
        return intHash64(static_cast<UInt64>(HashFunction{}(key))) % shards.size();
    }

    Shard & shardOf(const Key & key) { return *shards[shardIndex(key)]; }
    const Shard & shardOf(const Key & key) const { return *shards[shardIndex(key)]; }

    /// For each shard, the positions in `keys` of the keys that live in it.
    std::vector<std::vector<size_t>> positionsByShard(const std::vector<Key> & keys) const
    {
        std::vector<std::vector<size_t>> positions_by_shard(shards.size());
        for (size_t i = 0; i < keys.size(); ++i)
            positions_by_shard[shardIndex(keys[i])].push_back(i);
        return positions_by_shard;
    }

    template <typename Func>
    void forEachPolicy(Func && func)
    {
        for (auto & shard_ptr : shards)
        {
            Shard & shard = *shard_ptr;
            std::lock_guard lock(shard.mutex);
            func(*shard.cache_policy);
        }
    }

    template <typename Getter>
    size_t sumOverPolicies(Getter && getter) const
    {
        size_t res = 0;
        for (const auto & shard_ptr : shards)
        {
            const Shard & shard = *shard_ptr;
            std::lock_guard lock(shard.mutex);
            res += getter(*shard.cache_policy);
        }
        return res;
    }

    typename CachePolicy::OnRemoveEntryFunction makeOnRemoveEntryFunction()
    {
        return [this](size_t weight_loss, const MappedPtr & mapped_ptr)
        {
            onEntryRemoval(weight_loss, mapped_ptr);
        };
    }

    /// Represents pending insertion attempt.
    struct InsertToken
    {
        explicit InsertToken(Shard & shard_) : shard(shard_) {}

        std::mutex mutex;
        bool cleaned_up TSA_GUARDED_BY(mutex) = false;
        MappedPtr value TSA_GUARDED_BY(mutex);

        Shard & shard;
        size_t refcount = 0; /// Protected by the mutex of the shard
    };

    /// This class is responsible for removing used insert tokens from the insert_tokens map of the shard.
    /// Among several concurrent threads the first successful one is responsible for removal. But if they all
    /// fail, then the last one is responsible.
    struct InsertTokenHolder
    {
        const Key * key = nullptr;
        std::shared_ptr<InsertToken> token;
        bool cleaned_up = false;

        InsertTokenHolder() = default;

        void acquire(const Key * key_, const std::shared_ptr<InsertToken> & token_, std::lock_guard<std::mutex> & /* cache_lock */)
            TSA_NO_THREAD_SAFETY_ANALYSIS // disabled only because we can't reference the mutex of the shard from here
        {
            key = key_;
            token = token_;
            ++token->refcount;
        }

        void cleanup(std::lock_guard<std::mutex> & /* token_lock */, std::lock_guard<std::mutex> & /* cache_lock */)
            TSA_NO_THREAD_SAFETY_ANALYSIS // disabled only because we can't reference the mutex of the shard from here
        {
            token->shard.insert_tokens.erase(*key);
            token->cleaned_up = true;
            cleaned_up = true;
        }

        ~InsertTokenHolder()
        {
            if (!token)
                return;

            if (cleaned_up)
                return;

            std::lock_guard token_lock(token->mutex);

            if (token->cleaned_up)
                return;

            std::lock_guard cache_lock(token->shard.mutex);

            --token->refcount;
            if (token->refcount == 0)
                cleanup(token_lock, cache_lock);
        }
    };

    friend struct InsertTokenHolder;

    /// This is called when an entry is being evicted from the cache.
    /// Override this method if you want to handle individual entry removals from cache
    virtual void onEntryRemoval(size_t /*weight_loss*/, const MappedPtr &) { }
};


}
