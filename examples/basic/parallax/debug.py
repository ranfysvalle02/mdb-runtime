#!/usr/bin/env python3
"""
Debug utilities for Parallax - Error tracking, logging, and debugging tools
"""
import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import deque
import functools

# Debug mode from environment
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Setup structured logging
logger = logging.getLogger("Parallax.Debug")

# In-memory error log (last 100 errors)
error_log: deque = deque(maxlen=100)
performance_log: deque = deque(maxlen=50)


class ErrorTracker:
    """Track and store errors for debugging"""
    
    @staticmethod
    def log_error(
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """Log an error with full context"""
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
            "call_id": call_id,
            "stage": stage,
        }
        error_log.append(error_entry)
        
        # Also log to standard logger
        logger.error(
            f"Error in {stage or 'unknown'} (call_id: {call_id}): {error}",
            exc_info=True,
            extra={"context": context, "call_id": call_id, "stage": stage}
        )
        
        return error_entry
    
    @staticmethod
    def get_errors(limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent errors"""
        return list(error_log)[-limit:]
    
    @staticmethod
    def clear_errors():
        """Clear error log"""
        error_log.clear()


class PerformanceTracker:
    """Track performance metrics"""
    
    @staticmethod
    def log_timing(
        operation: str,
        duration: float,
        call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log a performance metric"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "duration_ms": duration * 1000,
            "call_id": call_id,
            "metadata": metadata or {},
        }
        performance_log.append(entry)
        
        if DEBUG_MODE:
            logger.debug(f"⏱️ {operation}: {duration*1000:.2f}ms (call_id: {call_id})")
        
        return entry
    
    @staticmethod
    def get_metrics(limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent performance metrics"""
        return list(performance_log)[-limit:]
    
    @staticmethod
    def clear_metrics():
        """Clear performance log"""
        performance_log.clear()


def debug_log(func):
    """Decorator to add debug logging to functions"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        if DEBUG_MODE:
            logger.debug(f"🔵 Calling {func.__name__} with args={args[:2]}, kwargs={list(kwargs.keys())}")
        try:
            start = asyncio.get_event_loop().time()
            result = await func(*args, **kwargs)
            duration = asyncio.get_event_loop().time() - start
            if DEBUG_MODE:
                logger.debug(f"✅ {func.__name__} completed in {duration*1000:.2f}ms")
            PerformanceTracker.log_timing(func.__name__, duration)
            return result
        except Exception as e:
            ErrorTracker.log_error(e, context={"function": func.__name__, "args": str(args[:2])})
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        if DEBUG_MODE:
            logger.debug(f"🔵 Calling {func.__name__} with args={args[:2]}, kwargs={list(kwargs.keys())}")
        try:
            import time
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            if DEBUG_MODE:
                logger.debug(f"✅ {func.__name__} completed in {duration*1000:.2f}ms")
            PerformanceTracker.log_timing(func.__name__, duration)
            return result
        except Exception as e:
            ErrorTracker.log_error(e, context={"function": func.__name__, "args": str(args[:2])})
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def get_debug_info() -> Dict[str, Any]:
    """Get comprehensive debug information"""
    return {
        "debug_mode": DEBUG_MODE,
        "log_level": LOG_LEVEL,
        "errors": {
            "count": len(error_log),
            "recent": ErrorTracker.get_errors(10),
        },
        "performance": {
            "count": len(performance_log),
            "recent": PerformanceTracker.get_metrics(10),
        },
        "environment": {
            "python_version": os.sys.version,
            "env_vars": {
                k: "***" if "key" in k.lower() or "secret" in k.lower() or "password" in k.lower() else v
                for k, v in os.environ.items()
                if k.startswith(("AZURE_", "OPENAI_", "MONGO_", "APP_", "LOG_", "DEBUG"))
            },
        },
    }
