-- The number of shards of the mark cache is a startup setting. Its default is one shard, so a server that does not
-- configure it keeps one mark cache with one lock, as before the setting existed.
SELECT name, value, changed FROM system.server_settings WHERE name = 'mark_cache_shards';
