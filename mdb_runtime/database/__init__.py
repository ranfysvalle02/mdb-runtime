"""
Database abstraction layer.

Provides scoped database access with automatic experiment isolation
and MongoDB-style API for familiarity.
"""

from .scoped_wrapper import (
    ScopedMongoWrapper,
    ScopedCollectionWrapper,
    AsyncAtlasIndexManager,
    AutoIndexManager,
)
from .abstraction import (
    ExperimentDB,
    Collection,
    get_experiment_db,
    create_actor_database,
)
from .connection import (
    get_shared_mongo_client,
    verify_shared_client,
    get_pool_metrics,
    close_shared_client,
)

__all__ = [
    # Scoped wrappers
    "ScopedMongoWrapper",
    "ScopedCollectionWrapper",
    "AsyncAtlasIndexManager",
    "AutoIndexManager",
    
    # Database abstraction
    "ExperimentDB",
    "Collection",
    "get_experiment_db",
    "create_actor_database",
    
    # Connection pooling
    "get_shared_mongo_client",
    "verify_shared_client",
    "get_pool_metrics",
    "close_shared_client",
]
