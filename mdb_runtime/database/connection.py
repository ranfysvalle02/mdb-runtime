"""
Shared MongoDB Connection Pool Manager

Provides a singleton MongoDB connection pool manager for Ray actors to share
database connections efficiently. This prevents each actor from creating its
own connection pool, reducing total connections from N×50 to N×5.

This module implements a thread-safe singleton pattern that ensures all Ray
actors in the same process share a single MongoDB client instance with a
reasonable connection pool size.

This module is part of MDB_RUNTIME - MongoDB Multi-Tenant Runtime Engine.

Usage:
    from mdb_runtime.database import get_shared_mongo_client, get_pool_metrics
    
    client = get_shared_mongo_client(mongo_uri)
    db = client[db_name]
    metrics = await get_pool_metrics()  # Monitor pool usage
"""

import asyncio
import logging
import os
import threading
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

# Global singleton instance
_shared_client: Optional[AsyncIOMotorClient] = None
# Use threading.Lock for cross-thread safety in Ray's actor environment
# Ray actors may run in different threads, so asyncio.Lock isn't sufficient
_init_lock = threading.Lock()


def get_shared_mongo_client(
    mongo_uri: str,
    max_pool_size: Optional[int] = None,
    min_pool_size: Optional[int] = None,
    server_selection_timeout_ms: int = 5000,
    max_idle_time_ms: int = 45000,
    retry_writes: bool = True,
    retry_reads: bool = True,
) -> AsyncIOMotorClient:
    """
    Gets or creates a shared MongoDB client instance.
    
    This function implements a singleton pattern to ensure all Ray actors
    in the same process share a single MongoDB client connection pool.
    
    Args:
        mongo_uri: MongoDB connection URI
        max_pool_size: Maximum connection pool size (default: from env or 10 for actors)
        min_pool_size: Minimum connection pool size (default: from env or 1)
        server_selection_timeout_ms: Server selection timeout in milliseconds
        max_idle_time_ms: Maximum idle time before closing connections
        retry_writes: Enable automatic retry for write operations
        retry_reads: Enable automatic retry for read operations
    
    Returns:
        Shared AsyncIOMotorClient instance
    
    Example:
        client = get_shared_mongo_client("mongodb://mongo:27017/")
        db = client["my_database"]
    """
    global _shared_client
    
    # Get pool sizes from environment or use defaults
    if max_pool_size is None:
        max_pool_size = int(os.getenv("MONGO_ACTOR_MAX_POOL_SIZE", "10"))
    if min_pool_size is None:
        min_pool_size = int(os.getenv("MONGO_ACTOR_MIN_POOL_SIZE", "1"))
    
    logger.info(
        f"MongoDB Actor Pool Configuration: max_pool_size={max_pool_size}, "
        f"min_pool_size={min_pool_size} (from env or defaults)"
    )
    
    # Fast path: return existing client if already initialized
    if _shared_client is not None:
        # Verify client is still connected
        try:
            # Non-blocking check - if client was closed, it will be None or invalid
            if hasattr(_shared_client, '_topology') and _shared_client._topology is not None:
                return _shared_client
        except Exception:
            # Client was closed or invalid, reset and recreate
            _shared_client = None
    
    # Thread-safe initialization with lock
    # Use threading lock for Ray's multi-threaded actor environment
    with _init_lock:
        # Double-check pattern: another thread may have initialized while we waited
        if _shared_client is not None:
            try:
                if hasattr(_shared_client, '_topology') and _shared_client._topology is not None:
                    return _shared_client
            except Exception:
                # Client was closed or invalid, reset and recreate
                _shared_client = None
        
        logger.info(
            f"Creating shared MongoDB client with pool_size={max_pool_size}, "
            f"min_pool_size={min_pool_size} (singleton for all actors in this process)"
        )
        
        try:
            _shared_client = AsyncIOMotorClient(
                mongo_uri,
                serverSelectionTimeoutMS=server_selection_timeout_ms,
                appname="ModularLabsActor",
                maxPoolSize=max_pool_size,
                minPoolSize=min_pool_size,
                maxIdleTimeMS=max_idle_time_ms,
                retryWrites=retry_writes,
                retryReads=retry_reads,
            )
            
            logger.info(
                f"Shared MongoDB client created successfully "
                f"(max_pool_size={max_pool_size}, min_pool_size={min_pool_size})"
            )
            logger.info(
                f"Connection pool limits: max={max_pool_size}, min={min_pool_size}. "
                f"Monitor pool usage with get_pool_metrics() to prevent exhaustion."
            )
            
            # Note: Ping verification happens asynchronously on first use
            # This avoids blocking during synchronous __init__ in Ray actors
            
            return _shared_client
        except Exception as e:
            logger.error(f"Failed to create shared MongoDB client: {e}", exc_info=True)
            _shared_client = None
            raise
    
    return _shared_client




async def verify_shared_client() -> bool:
    """
    Verifies that the shared MongoDB client is connected.
    Should be called from async context after initialization.
    
    Returns:
        True if client is connected and responsive, False otherwise
    """
    global _shared_client
    
    if _shared_client is None:
        logger.warning("Shared MongoDB client is None - cannot verify")
        return False
    
    try:
        await _shared_client.admin.command("ping")
        logger.debug("Shared MongoDB client verification successful")
        return True
    except Exception as e:
        logger.error(f"Shared MongoDB client verification failed: {e}")
        return False


async def get_pool_metrics() -> Dict[str, Any]:
    """
    Gets connection pool metrics for monitoring.
    Returns information about pool size, active connections, etc.
    
    Returns:
        Dictionary with pool metrics including:
        - max_pool_size: Maximum pool size
        - min_pool_size: Minimum pool size
        - active_connections: Current active connections (if available)
        - pool_usage_percent: Estimated pool usage percentage
    """
    global _shared_client
    
    if _shared_client is None:
        return {
            "status": "no_client",
            "error": "Shared MongoDB client not initialized"
        }
    
    try:
        # Get topology information if available
        topology = getattr(_shared_client, '_topology', None)
        pool_opts = getattr(_shared_client, '_pool_options', {})
        
        max_pool_size = pool_opts.get('max_pool_size', getattr(_shared_client, 'max_pool_size', None))
        min_pool_size = pool_opts.get('min_pool_size', getattr(_shared_client, 'min_pool_size', None))
        
        # Try to get active connection count from topology
        active_connections = None
        if topology and hasattr(topology, 'servers'):
            # Count connections across all servers
            total_connections = 0
            for server in topology.servers.values():
                if hasattr(server, 'pool') and server.pool:
                    if hasattr(server.pool, 'checked_out'):
                        total_connections += len(server.pool.checked_out)
            
            active_connections = total_connections if total_connections > 0 else None
        
        metrics = {
            "status": "connected",
            "max_pool_size": max_pool_size,
            "min_pool_size": min_pool_size,
            "active_connections": active_connections,
        }
        
        # Calculate usage percentage if we have both values
        if max_pool_size and active_connections is not None:
            usage_percent = (active_connections / max_pool_size) * 100
            metrics["pool_usage_percent"] = round(usage_percent, 2)
            
            # Warn if pool usage is high
            if usage_percent > 80:
                logger.warning(
                    f"MongoDB connection pool usage is HIGH: {usage_percent:.1f}% "
                    f"({active_connections}/{max_pool_size}). Consider increasing max_pool_size."
                )
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting pool metrics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def close_shared_client():
    """
    Closes the shared MongoDB client.
    Should be called during application shutdown.
    """
    global _shared_client
    
    if _shared_client is not None:
        try:
            _shared_client.close()
            logger.info("Shared MongoDB client closed")
        except Exception as e:
            logger.warning(f"Error closing shared MongoDB client: {e}")
        finally:
            _shared_client = None

