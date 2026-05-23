use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::{error::FieldError, field::Field};

struct CacheState {
    map: HashMap<[u8; 32], Arc<Field>>,
    order: VecDeque<[u8; 32]>,
}

/// LRU in-memory cache keyed by `creation_hash`.
pub struct FieldCache {
    inner: Mutex<CacheState>,
    capacity: usize,
}

impl FieldCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(CacheState {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            capacity,
        }
    }

    pub fn get(&self, key: &[u8; 32]) -> Option<Arc<Field>> {
        let mut guard = self.inner.lock().ok()?;
        let value = guard.map.get(key).cloned()?;
        guard.order.retain(|existing| existing != key);
        guard.order.push_back(*key);
        Some(value)
    }

    pub fn insert(&self, field: Arc<Field>) {
        let Some(key) = field.metadata().creation_hash() else {
            return;
        };
        if self.capacity == 0 {
            return;
        }

        let Ok(mut guard) = self.inner.lock() else {
            return;
        };

        use std::collections::hash_map::Entry;
        match guard.map.entry(key) {
            Entry::Occupied(mut e) => {
                e.insert(field);
                guard.order.retain(|existing| *existing != key);
                guard.order.push_back(key);
                return;
            }
            Entry::Vacant(e) => {
                e.insert(field);
                guard.order.push_back(key);
            }
        }

        while guard.map.len() > self.capacity {
            if let Some(evict_key) = guard.order.pop_front() {
                guard.map.remove(&evict_key);
            } else {
                break;
            }
        }
    }
}

/// Computes a field and optionally interns it into the cache.
pub fn cached_or_compute<F>(
    cache: Option<&FieldCache>,
    compute: F,
) -> Result<Arc<Field>, FieldError>
where
    F: FnOnce() -> Result<Field, FieldError>,
{
    let field = compute()?;
    let Some(key) = field.metadata().creation_hash() else {
        return Err(FieldError::InvalidMetadata);
    };

    if let Some(cache_ref) = cache {
        if let Some(existing) = cache_ref.get(&key) {
            return Ok(existing);
        }

        let field_arc = Arc::new(field);
        cache_ref.insert(Arc::clone(&field_arc));
        return Ok(field_arc);
    }

    Ok(Arc::new(field))
}
