#!/usr/bin/env python3
"""
SSO App 2 - Alpaca Paper Trading (Virtual Ledger)
==================================================

SSO-enabled app for Alpaca paper trading using Trade API with Virtual Ledger.
Uses a single master Alpaca account and tracks user balances/positions in MongoDB.
Users can register virtual accounts instantly and execute trades through the master account.
"""

import asyncio
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from mdb_engine.dependencies import get_scoped_db
from mdb_engine.utils import clean_mongo_doc, clean_mongo_docs

# Configure logging first
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning(f".env file not found at {env_path}")
except ImportError:
    # python-dotenv not installed, skip .env loading
    logger.warning(
        "python-dotenv not installed. .env file will not be loaded. "
        "Install with: pip install python-dotenv"
    )

# Alpaca SDK imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca SDK not available. Install with: pip install alpaca-py")

try:
    from mdb_engine import MongoDBEngine
except ImportError as e:
    logger.error(f"Failed to import mdb_engine: {e}", exc_info=True)
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_SLUG = "sso-app-2"

# Initialize MongoDB Engine
try:
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = os.getenv("MONGODB_DB", "oblivio_apps")
    logger.info(f"Initializing MongoDBEngine with URI: {mongo_uri[:50]}... (db: {db_name})")

    engine = MongoDBEngine(
        mongo_uri=mongo_uri,
        db_name=db_name,
    )
    logger.info("MongoDBEngine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MongoDBEngine: {e}", exc_info=True)
    sys.exit(1)

# Create FastAPI app
try:
    manifest_path = Path(__file__).parent / "manifest.json"
    logger.info(f"Creating FastAPI app with manifest: {manifest_path}")

    app = engine.create_app(
        slug=APP_SLUG,
        manifest=manifest_path,
        title="SSO App 2",
        description="Alpaca Paper Trading SSO app",
        version="1.0.0",
    )
    logger.info("FastAPI app created successfully")
except Exception as e:
    logger.error(f"Failed to create FastAPI app: {e}", exc_info=True)
    sys.exit(1)

# Template engine
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ============================================================================
# ALPACA CONFIGURATION
# ============================================================================

# Alpaca API credentials from environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# Log Alpaca configuration status
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    logger.info("Alpaca API credentials loaded from environment")
else:
    logger.warning(
        f"Alpaca API credentials not configured. "
        f"ALPACA_API_KEY={'set' if ALPACA_API_KEY else 'NOT SET'}, "
        f"ALPACA_SECRET_KEY={'set' if ALPACA_SECRET_KEY else 'NOT SET'}. "
        f"Set these in .env file or environment variables."
    )

# Initialize Alpaca Trading Client (Paper Trading Mode)
trading_client = None
if ALPACA_AVAILABLE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
    try:
        trading_client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True,  # Paper trading mode
        )
        # Test connection by getting account info
        try:
            account = trading_client.get_account()
            logger.info(
                f"Alpaca Trading Client initialized (paper trading mode) - "
                f"Account equity: ${account.equity}"
            )
        except Exception as test_error:
            logger.exception(
                f"Alpaca Trading Client initialized but connection test failed: {test_error}"
            )
            logger.warning(
                "This usually means the API credentials are invalid or "
                "don't have Trade API access."
            )
            logger.warning("Note: Broker API credentials are different from Trade API credentials.")
            logger.warning("Make sure you're using Trade API credentials (paper trading).")
            # Don't set trading_client to None - let it fail later with better error messages
    except Exception as e:
        logger.exception(f"Failed to initialize Alpaca Trading Client: {e}")
        logger.warning("Check that ALPACA_API_KEY and ALPACA_SECRET_KEY are set correctly.")
        logger.warning("For paper trading, use Trade API credentials (not Broker API credentials).")
elif not ALPACA_AVAILABLE:
    logger.warning("Alpaca SDK not available. Paper trading features will be disabled.")
elif not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.warning(
        "Alpaca API credentials not configured. "
        "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
    )
    logger.warning(
        "For paper trading, you need Trade API credentials from https://app.alpaca.markets/paper/dashboard/overview"
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_auth_hub_url() -> str:
    """
    Get auth hub URL from manifest, environment variable, or default.

    Priority:
    1. manifest.auth.auth_hub_url (if mode is "shared")
    2. AUTH_HUB_URL environment variable
    3. Default: http://localhost:8000

    Returns:
        Auth hub URL string
    """
    # Try manifest first
    manifest = getattr(app.state, "manifest", None)
    if manifest:
        auth_config = manifest.get("auth", {})
        if auth_config.get("mode") == "shared":
            auth_hub_url = auth_config.get("auth_hub_url")
            if auth_hub_url:
                return auth_hub_url

    # Fallback to environment variable
    return os.getenv("AUTH_HUB_URL", "http://localhost:8000")


def get_current_user(request: Request) -> dict | None:
    """Get user from request.state (populated by SharedAuthMiddleware)."""
    return getattr(request.state, "user", None)


def get_roles(user: dict | None) -> list:
    """Get user's roles for this app."""
    if not user:
        return []
    app_roles = user.get("app_roles", {})
    return app_roles.get(APP_SLUG, [])


def is_admin(user: dict | None) -> bool:
    """Check if user is admin."""
    roles = get_roles(user)
    return "admin" in roles


# ============================================================================
# VIRTUAL LEDGER FUNCTIONS
# ============================================================================

# Rate limiting: Track order submissions (200 req/min for Paper)
_order_submission_times = []
_order_lock = asyncio.Lock()


async def initialize_user_balance(
    user_email: str, initial_cash: float = 1000.0, db=None
) -> dict | None:
    """
    Create a virtual trading account for a user.

    Args:
        user_email: User's email address
        initial_cash: Initial cash balance (default $1000)
        db: Scoped database wrapper

    Returns:
        Dictionary with balance data, or None if failed
    """
    if not db:
        logger.error("Database not provided")
        return None

    try:
        buying_power = initial_cash * 2.0  # 2x margin for paper trading

        balance_doc = {
            "user_email": user_email,
            "virtual_cash": initial_cash,
            "buying_power": buying_power,
            "equity": initial_cash,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }

        # Use upsert to handle race conditions
        await db.user_balances.update_one(
            {"user_email": user_email}, {"$setOnInsert": balance_doc}, upsert=True
        )

        logger.info(f"Initialized virtual balance for {user_email}: ${initial_cash}")
        return balance_doc
    except Exception as e:
        logger.error(f"Failed to initialize user balance for {user_email}: {e}", exc_info=True)
        return None


async def get_user_balance(user_email: str, db) -> dict | None:
    """
    Get user's virtual balance from MongoDB.

    Args:
        user_email: User's email address
        db: Scoped database wrapper

    Returns:
        Dictionary with balance data, or None if not found
    """
    balance = await db.user_balances.find_one({"user_email": user_email})
    return balance


async def update_user_cash(user_email: str, amount: float, db) -> bool:
    """
    Update user's virtual cash balance.

    Args:
        user_email: User's email address
        amount: Amount to add (positive) or subtract (negative)
        db: Scoped database wrapper

    Returns:
        True if successful, False otherwise
    """
    try:
        result = await db.user_balances.update_one(
            {"user_email": user_email},
            {"$inc": {"virtual_cash": amount}, "$set": {"updated_at": datetime.now(timezone.utc)}},
        )

        if result.modified_count > 0:
            # Recalculate buying power and equity
            balance = await get_user_balance(user_email, db)
            if balance:
                new_cash = balance.get("virtual_cash", 0)
                buying_power = new_cash * 2.0  # 2x margin

                # Calculate equity from positions
                positions = await get_user_positions(user_email, db)
                position_value = sum(p.get("cost_basis", 0) for p in positions)
                equity = new_cash + position_value

                await db.user_balances.update_one(
                    {"user_email": user_email},
                    {
                        "$set": {
                            "buying_power": buying_power,
                            "equity": equity,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )

            logger.info(f"Updated cash for {user_email}: ${amount}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to update cash for {user_email}: {e}", exc_info=True)
        return False


async def get_user_positions(user_email: str, db) -> list:
    """
    Get all positions for a user.
    Wrapper around get_user_scoped_positions for backward compatibility.

    Args:
        user_email: User's email address
        db: Scoped database wrapper

    Returns:
        List of position dictionaries
    """
    return await get_user_scoped_positions(user_email, db)


# ============================================================================
# POSITION HELPERS (User-Scoped)
# ============================================================================


async def get_user_scoped_positions(user_email: str, db, **filters) -> list:
    """
    Get user positions with additional filters. Always scoped to user_email.

    Args:
        user_email: User's email address
        db: Scoped database wrapper
        **filters: Additional MongoDB filters to apply

    Returns:
        List of position dictionaries
    """
    query = {"user_email": user_email, **filters}
    positions = await db.user_positions.find(query).to_list(length=None)
    return positions or []


async def get_user_position_by_symbol(user_email: str, symbol: str, db) -> dict | None:
    """
    Get a specific position for a user by symbol.

    Args:
        user_email: User's email address
        symbol: Stock symbol
        db: Scoped database wrapper

    Returns:
        Position dictionary or None if not found
    """
    position = await db.user_positions.find_one({"user_email": user_email, "symbol": symbol})
    return position


async def has_user_position(user_email: str, symbol: str, db) -> bool:
    """
    Check if user has a position for a symbol.

    Args:
        user_email: User's email address
        symbol: Stock symbol
        db: Scoped database wrapper

    Returns:
        True if position exists, False otherwise
    """
    position = await db.user_positions.find_one({"user_email": user_email, "symbol": symbol})
    return position is not None


async def get_user_positions_by_symbols(user_email: str, symbols: list[str], db) -> list:
    """
    Get positions for multiple symbols.

    Args:
        user_email: User's email address
        symbols: List of stock symbols
        db: Scoped database wrapper

    Returns:
        List of position dictionaries
    """
    if not symbols:
        return []
    positions = await db.user_positions.find(
        {"user_email": user_email, "symbol": {"$in": symbols}}
    ).to_list(length=None)
    return positions or []


async def add_user_position(
    user_email: str, symbol: str, shares: float, avg_price: float, db
) -> bool:
    """
    Add or update a user's position.

    Args:
        user_email: User's email address
        symbol: Stock symbol
        shares: Number of shares (positive for long, negative for short)
        avg_price: Average price per share
        db: Scoped database wrapper

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get existing position
        existing = await db.user_positions.find_one({"user_email": user_email, "symbol": symbol})

        if existing:
            # Update existing position
            old_shares = existing.get("shares", 0)
            old_cost = existing.get("cost_basis", 0)

            new_shares = old_shares + shares

            if shares < 0:
                # Selling: reduce cost basis proportionally (FIFO)
                if old_shares > 0:
                    reduction_ratio = abs(shares) / old_shares
                    new_cost = old_cost * (1 - reduction_ratio)
                else:
                    new_cost = old_cost  # Can't reduce more than we have
            else:
                # Buying: add to cost basis (weighted average)
                new_cost = old_cost + (shares * avg_price)

            if new_shares == 0:
                # Position closed, remove it
                await db.user_positions.delete_one({"user_email": user_email, "symbol": symbol})
            else:
                # Recalculate average price from cost basis
                new_avg = new_cost / new_shares if new_shares != 0 else 0

                await db.user_positions.update_one(
                    {"user_email": user_email, "symbol": symbol},
                    {
                        "$set": {
                            "shares": new_shares,
                            "avg_price": new_avg,
                            "cost_basis": new_cost,
                            "updated_at": datetime.now(timezone.utc),
                        }
                    },
                )
        else:
            # Create new position
            if shares != 0:
                await db.user_positions.insert_one(
                    {
                        "user_email": user_email,
                        "symbol": symbol,
                        "shares": shares,
                        "avg_price": avg_price,
                        "cost_basis": shares * avg_price,
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

        # Update equity in balance
        balance = await get_user_balance(user_email, db)
        if balance:
            positions = await get_user_positions(user_email, db)
            position_value = sum(p.get("cost_basis", 0) for p in positions)
            equity = balance.get("virtual_cash", 0) + position_value

            await db.user_balances.update_one(
                {"user_email": user_email},
                {"$set": {"equity": equity, "updated_at": datetime.now(timezone.utc)}},
            )

        logger.info(f"Updated position for {user_email}: {shares:+} {symbol} @ ${avg_price}")
        return True
    except Exception as e:
        logger.error(f"Failed to add position for {user_email}: {e}", exc_info=True)
        return False


async def get_user_portfolio_value(user_email: str, db) -> dict:
    """
    Calculate total portfolio value for a user.

    Args:
        user_email: User's email address
        db: Scoped database wrapper

    Returns:
        Dictionary with portfolio summary
    """
    balance = await get_user_balance(user_email, db)
    positions = await get_user_positions(user_email, db)

    cash = balance.get("virtual_cash", 0) if balance else 0
    position_value = sum(p.get("cost_basis", 0) for p in positions)
    equity = cash + position_value

    return {
        "cash": cash,
        "positions_value": position_value,
        "equity": equity,
        "buying_power": cash * 2.0,  # 2x margin
        "position_count": len(positions),
    }


async def get_master_account_info() -> dict | None:
    """
    Get master Alpaca account info (for rate limiting awareness).

    Returns:
        Dictionary with account info, or None if failed
    """
    if not trading_client:
        return None

    try:
        account = trading_client.get_account()
        return {
            "equity": float(account.equity) if account.equity else 0,
            "buying_power": float(account.buying_power) if account.buying_power else 0,
            "cash": float(account.cash) if account.cash else 0,
            "pattern_day_trader": account.pattern_day_trader
            if hasattr(account, "pattern_day_trader")
            else False,
        }
    except Exception as e:
        logger.error(f"Failed to get master account info: {e}", exc_info=True)
        return None


# ============================================================================
# ORDER HELPERS (User-Scoped + Smart Queries)
# ============================================================================


async def get_user_orders(
    user_email: str, db, limit: int = 100, sort_by: str = "submitted_at", sort_order: int = -1
) -> list:
    """
    Get all orders for a user. Base helper that always scopes to user_email.

    Args:
        user_email: User's email address
        db: Scoped database wrapper
        limit: Maximum number of orders to return
        sort_by: Field to sort by
        sort_order: Sort order (-1 for descending, 1 for ascending)

    Returns:
        List of order dictionaries
    """
    orders = (
        await db.user_orders.find({"user_email": user_email})
        .sort(sort_by, sort_order)
        .to_list(length=limit)
    )
    return orders or []


async def get_user_order_by_id(user_email: str, order_id: str, db) -> dict | None:
    """
    Find order by Alpaca ID or client ID. Always scoped to user_email.

    Args:
        user_email: User's email address
        order_id: Alpaca order ID or client order ID
        db: Scoped database wrapper

    Returns:
        Order dictionary or None if not found
    """
    order = await db.user_orders.find_one(
        {
            "user_email": user_email,
            "$or": [{"alpaca_order_id": order_id}, {"client_order_id": order_id}],
        }
    )
    return order


async def get_user_order_by_alpaca_id(user_email: str, alpaca_order_id: str, db) -> dict | None:
    """
    Find order by Alpaca order ID. Always scoped to user_email.

    Args:
        user_email: User's email address
        alpaca_order_id: Alpaca order ID
        db: Scoped database wrapper

    Returns:
        Order dictionary or None if not found
    """
    order = await db.user_orders.find_one(
        {"user_email": user_email, "alpaca_order_id": alpaca_order_id}
    )
    return order


async def get_user_orders_by_status(user_email: str, status: str, db, limit: int = 100) -> list:
    """
    Get orders filtered by status. Always scoped to user_email.

    Args:
        user_email: User's email address
        status: Order status (pending, filled, cancelled, etc.)
        db: Scoped database wrapper
        limit: Maximum number of orders to return

    Returns:
        List of order dictionaries
    """
    orders = (
        await db.user_orders.find({"user_email": user_email, "status": status})
        .sort("submitted_at", -1)
        .to_list(length=limit)
    )
    return orders or []


async def get_user_pending_orders(user_email: str, db) -> list:
    """
    Get all pending orders for a user. Includes all pending statuses.

    Args:
        user_email: User's email address
        db: Scoped database wrapper

    Returns:
        List of pending order dictionaries
    """
    pending_statuses = [
        "pending",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
        "pending_replace",
        "pending_cancel",
    ]
    orders = (
        await db.user_orders.find({"user_email": user_email, "status": {"$in": pending_statuses}})
        .sort("submitted_at", -1)
        .to_list(length=100)
    )
    return orders or []


async def get_user_orders_needing_refresh(user_email: str, db) -> list:
    """
    Get orders that need status refresh from Alpaca API.

    Args:
        user_email: User's email address
        db: Scoped database wrapper

    Returns:
        List of orders that need refresh (pending orders with alpaca_order_id)
    """
    pending_statuses = [
        "pending",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
        "pending_cancel",
    ]
    orders = (
        await db.user_orders.find(
            {
                "user_email": user_email,
                "status": {"$in": pending_statuses},
                "alpaca_order_id": {"$exists": True, "$ne": None},
            }
        )
        .sort("submitted_at", -1)
        .to_list(length=100)
    )
    return orders or []


async def get_user_orders_by_symbol(user_email: str, symbol: str, db, limit: int = 100) -> list:
    """
    Get all orders for a specific symbol. Always scoped to user_email.

    Args:
        user_email: User's email address
        symbol: Stock symbol
        db: Scoped database wrapper
        limit: Maximum number of orders to return

    Returns:
        List of order dictionaries
    """
    orders = (
        await db.user_orders.find({"user_email": user_email, "symbol": symbol})
        .sort("submitted_at", -1)
        .to_list(length=limit)
    )
    return orders or []


async def get_user_orders_by_symbol_and_side(
    user_email: str, symbol: str, side: str, db, limit: int = 100
) -> list:
    """
    Get orders by symbol and side. Always scoped to user_email.

    Args:
        user_email: User's email address
        symbol: Stock symbol
        side: Order side (buy or sell)
        db: Scoped database wrapper
        limit: Maximum number of orders to return

    Returns:
        List of order dictionaries
    """
    orders = (
        await db.user_orders.find({"user_email": user_email, "symbol": symbol, "side": side})
        .sort("submitted_at", -1)
        .to_list(length=limit)
    )
    return orders or []


async def has_user_pending_order(user_email: str, symbol: str, side: str, db) -> bool:
    """
    Check if user has a pending order for a symbol and side.

    Args:
        user_email: User's email address
        symbol: Stock symbol
        side: Order side (buy or sell)
        db: Scoped database wrapper

    Returns:
        True if pending order exists, False otherwise
    """
    pending_statuses = [
        "pending",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
        "pending_replace",
        "pending_cancel",
    ]
    order = await db.user_orders.find_one(
        {
            "user_email": user_email,
            "symbol": symbol,
            "side": side,
            "status": {"$in": pending_statuses},
        }
    )
    return order is not None


async def get_user_conflicting_orders(user_email: str, symbol: str, side: str, db) -> list:
    """
    Get conflicting orders (opposite side, same symbol, pending status).
    Used for wash trade detection.

    Args:
        user_email: User's email address
        symbol: Stock symbol
        side: Order side (buy or sell) - will find opposite side orders
        db: Scoped database wrapper

    Returns:
        List of conflicting order dictionaries
    """
    opposite_side = "sell" if side == "buy" else "buy"
    pending_statuses = [
        "pending",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
        "pending_replace",
        "pending_cancel",
    ]

    orders = (
        await db.user_orders.find(
            {
                "user_email": user_email,
                "symbol": symbol,
                "side": opposite_side,
                "status": {"$in": pending_statuses},
            }
        )
        .sort("submitted_at", -1)
        .to_list(length=10)
    )
    return orders or []


async def get_user_order_stats(user_email: str, db) -> dict:
    """
    Get order statistics for a user.

    Args:
        user_email: User's email address
        db: Scoped database wrapper

    Returns:
        Dictionary with order statistics
    """
    all_orders = await get_user_orders(user_email, db, limit=1000)

    stats = {
        "total": len(all_orders),
        "by_status": {},
        "by_symbol": {},
        "pending_count": 0,
        "filled_count": 0,
        "cancelled_count": 0,
    }

    for order in all_orders:
        status = order.get("status", "unknown")
        symbol = order.get("symbol", "unknown")

        # Count by status
        stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

        # Count by symbol
        stats["by_symbol"][symbol] = stats["by_symbol"].get(symbol, 0) + 1

        # Count specific statuses
        if status in [
            "pending",
            "accepted",
            "pending_new",
            "accepted_for_bidding",
            "pending_cancel",
        ]:
            stats["pending_count"] += 1
        elif status == "filled":
            stats["filled_count"] += 1
        elif status in ["cancelled", "expired", "rejected"]:
            stats["cancelled_count"] += 1

    return stats


# ============================================================================
# ORDER STATUS REFRESH HELPERS
# ============================================================================


async def refresh_order_status(user_email: str, alpaca_order_id: str, db) -> dict | None:
    """
    Refresh single order status from Alpaca API.

    Args:
        user_email: User's email address
        alpaca_order_id: Alpaca order ID
        db: Scoped database wrapper

    Returns:
        Updated order dictionary or None if failed
    """
    if not trading_client:
        return None

    try:
        # Get order from database first (scoped to user)
        order = await get_user_order_by_alpaca_id(user_email, alpaca_order_id, db)
        if not order:
            logger.debug(f"Order not found for refresh: {alpaca_order_id}")
            return None

        # Get latest status from Alpaca
        alpaca_order = trading_client.get_order_by_id(alpaca_order_id)
        if not alpaca_order:
            return None

        # Update order status in database
        update_data = {
            "status": alpaca_order.status
            if hasattr(alpaca_order, "status")
            else order.get("status"),
            "updated_at": datetime.now(timezone.utc),
        }

        if hasattr(alpaca_order, "filled_qty") and alpaca_order.filled_qty:
            update_data["filled_qty"] = float(alpaca_order.filled_qty)
        if hasattr(alpaca_order, "filled_avg_price") and alpaca_order.filled_avg_price:
            update_data["filled_avg_price"] = float(alpaca_order.filled_avg_price)

        await db.user_orders.update_one(
            {"user_email": user_email, "alpaca_order_id": alpaca_order_id}, {"$set": update_data}
        )

        # If order is filled, process it
        if hasattr(alpaca_order, "status") and alpaca_order.status == "filled":
            await handle_order_fill(alpaca_order_id, db)

        # Return updated order
        return await get_user_order_by_alpaca_id(user_email, alpaca_order_id, db)

    except (ValueError, KeyError, AttributeError, OSError, TimeoutError) as e:
        logger.debug(f"Could not refresh order {alpaca_order_id}: {e}", exc_info=True)
        return None


async def refresh_user_pending_orders(user_email: str, db) -> dict:
    """
    Batch refresh all pending orders for a user.

    Args:
        user_email: User's email address
        db: Scoped database wrapper

    Returns:
        Dictionary with refresh results
    """
    orders_to_refresh = await get_user_orders_needing_refresh(user_email, db)

    results = {"total": len(orders_to_refresh), "refreshed": 0, "failed": 0, "filled": 0}

    for order in orders_to_refresh:
        alpaca_order_id = order.get("alpaca_order_id")
        if not alpaca_order_id:
            continue

        try:
            updated_order = await refresh_order_status(user_email, alpaca_order_id, db)
            if updated_order:
                results["refreshed"] += 1
                if updated_order.get("status") == "filled":
                    results["filled"] += 1
            else:
                results["failed"] += 1
        except (ValueError, KeyError, AttributeError, OSError, TimeoutError) as e:
            logger.debug(f"Failed to refresh order {alpaca_order_id}: {e}", exc_info=True)
            results["failed"] += 1

    return results


async def auto_refresh_order_if_needed(order: dict, db) -> dict | None:
    """
    Check if order needs refresh and refresh if needed.

    Args:
        order: Order dictionary
        db: Scoped database wrapper

    Returns:
        Updated order dictionary or None if not refreshed
    """
    user_email = order.get("user_email")
    if not user_email:
        return None

    status = order.get("status", "").lower()
    pending_statuses = [
        "pending",
        "accepted",
        "pending_new",
        "accepted_for_bidding",
        "pending_cancel",
    ]

    if status not in pending_statuses:
        return None  # Order doesn't need refresh

    alpaca_order_id = order.get("alpaca_order_id")
    if not alpaca_order_id:
        return None  # No Alpaca ID to refresh

    return await refresh_order_status(user_email, alpaca_order_id, db)


async def execute_trade(
    user_email: str,
    symbol: str,
    side: str,
    qty: float,
    order_type: str = "market",
    limit_price: float = None,
    db=None,
) -> dict:
    """
    Execute a trade for a user through the master account.

    Args:
        user_email: User's email address
        symbol: Stock symbol
        side: "buy" or "sell"
        qty: Number of shares
        order_type: "market" or "limit"
        limit_price: Limit price (required if order_type is "limit")
        db: Scoped database wrapper

    Returns:
        Dictionary with order result
    """
    if not trading_client:
        return {"success": False, "error": "Trading client not initialized"}

    if not db:
        return {"success": False, "error": "Database not provided"}

    # Validate order
    if side not in ["buy", "sell"]:
        return {"success": False, "error": "Invalid side. Must be 'buy' or 'sell'"}

    if order_type == "limit" and limit_price is None:
        return {"success": False, "error": "Limit price required for limit orders"}

    # Check buying power for buy orders
    if side == "buy":
        balance = await get_user_balance(user_email, db)
        if not balance:
            return {"success": False, "error": "User balance not found. Please register first."}

        # Estimate cost (use limit price if provided, otherwise assume
        # current price * 1.1 for safety)
        estimated_cost = qty * (limit_price if limit_price else 100.0) * 1.1

        if estimated_cost > balance.get("buying_power", 0):
            return {
                "success": False,
                "error": (
                    f"Insufficient buying power. Need ${estimated_cost:.2f}, "
                    f"have ${balance.get('buying_power', 0):.2f}"
                ),
            }

    # Check position for sell orders (including pending buy orders)
    if side == "sell":
        positions = await get_user_positions(user_email, db)
        user_position = next((p for p in positions if p.get("symbol") == symbol), None)
        current_shares = user_position.get("shares", 0) if user_position else 0

        # Check for pending buy orders for this symbol using helper
        pending_buys = await get_user_orders_by_symbol_and_side(
            user_email, symbol, "buy", db, limit=10
        )

        # Calculate total shares including pending buys
        pending_shares = sum(float(order.get("qty", 0)) for order in pending_buys)
        total_available_shares = current_shares + pending_shares

        if total_available_shares < qty:
            if pending_shares > 0:
                return {
                    "success": False,
                    "error": (
                        f"Insufficient shares. Trying to sell {qty}, have {current_shares} "
                        f"shares and {pending_shares} shares pending from buy orders. "
                        "Please wait for buy orders to fill or cancel them first."
                    ),
                }
            else:
                return {
                    "success": False,
                    "error": f"Insufficient shares. Trying to sell {qty}, have {current_shares}",
                }

    # Check for conflicting orders (wash trade prevention)
    conflicting_orders = await get_user_conflicting_orders(user_email, symbol, side, db)

    if conflicting_orders:
        conflicting_order = conflicting_orders[0]
        opposite_side = conflicting_order.get("side")
        conflicting_order_id = (
            conflicting_order.get("alpaca_order_id")
            or conflicting_order.get("client_order_id")
            or conflicting_order.get("_id")
        )

        # Convert datetime to ISO string for JSON serialization
        submitted_at = conflicting_order.get("submitted_at")
        if submitted_at and isinstance(submitted_at, datetime):
            submitted_at = submitted_at.isoformat()
        elif submitted_at:
            submitted_at = str(submitted_at)

        return {
            "success": False,
            "error": (
                f"Cannot place {side} order: You have a pending {opposite_side} order "
                f"for {symbol}. Please cancel the existing order first or wait for it to fill."
            ),
            "conflicting_order_id": str(conflicting_order_id),
            "conflicting_order": {
                "side": conflicting_order.get("side"),
                "qty": conflicting_order.get("qty"),
                "status": conflicting_order.get("status"),
                "submitted_at": submitted_at,
            },
        }

    # Rate limiting check
    async with _order_lock:
        current_time = datetime.now(timezone.utc).timestamp()
        # Remove orders older than 1 minute
        _order_submission_times[:] = [t for t in _order_submission_times if current_time - t < 60]

        if len(_order_submission_times) >= 200:
            return {"success": False, "error": "Rate limit exceeded. Please try again in a moment."}

        _order_submission_times.append(current_time)

    try:
        # Verify trading client is initialized
        if not trading_client:
            return {
                "success": False,
                "error": (
                    "Trading client not initialized. "
                    "Check server logs for API credential issues."
                ),
            }

        # Test connection first
        try:
            test_account = trading_client.get_account()
            logger.debug(f"Connection test successful - Account equity: ${test_account.equity}")
        except Exception as conn_error:
            error_msg = str(conn_error)
            if "unauthorized" in error_msg.lower() or "401" in error_msg:
                logger.exception("API credentials are invalid or unauthorized.")
                logger.warning(
                    "Make sure you're using Trade API credentials " "(not Broker API credentials)."
                )
                logger.warning(
                    "Get your paper trading API keys from: "
                    "https://app.alpaca.markets/paper/dashboard/overview"
                )
                return {
                    "success": False,
                    "error": (
                        "API credentials are invalid. Please check your "
                        "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables. "
                        "Make sure you're using Trade API credentials for paper trading."
                    ),
                }
            else:
                logger.exception(f"Connection test failed: {conn_error}")
                return {
                    "success": False,
                    "error": f"Failed to connect to Alpaca API: {str(conn_error)}",
                }

        # Create client_order_id for tracking
        import time

        client_order_id = f"{user_email}_{int(time.time())}_{random.randint(1000, 9999)}"

        # Create order request
        if order_type == "market":
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
        else:  # limit
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )

        # Submit order to Alpaca
        try:
            order = trading_client.submit_order(order_request)
        except Exception as submit_error:
            # Handle wash trade and other Alpaca API errors
            error_str = str(submit_error).lower()
            error_msg = str(submit_error)

            # Try to extract error details from exception
            existing_order_id = None
            error_code = None

            # Check various exception attributes for error details
            if hasattr(submit_error, "status_code"):
                error_code = submit_error.status_code
            if hasattr(submit_error, "code"):
                error_code = submit_error.code
            if hasattr(submit_error, "body"):
                try:
                    error_body = (
                        json.loads(submit_error.body)
                        if isinstance(submit_error.body, str)
                        else submit_error.body
                    )
                    existing_order_id = error_body.get("existing_order_id")
                    if not error_code:
                        error_code = error_body.get("code")
                    if not error_msg or error_msg == str(submit_error):
                        error_msg = error_body.get("message", error_msg)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    # Ignore parsing errors - we'll use the original error message
                    pass

            # Check if it's a wash trade error (40310000 or error message contains wash trade)
            is_wash_trade = (
                str(error_code) == "40310000"
                or "40310000" in error_msg
                or "wash trade" in error_str
                or "opposite side" in error_str
                or "opposite side market" in error_str
            )

            if is_wash_trade:
                # Find the conflicting order in our database
                conflicting_orders = await get_user_conflicting_orders(user_email, symbol, side, db)

                conflicting_order_id = None
                opposite_side = "sell" if side == "buy" else "buy"
                if conflicting_orders:
                    conflicting_order_id = conflicting_orders[0].get(
                        "alpaca_order_id"
                    ) or conflicting_orders[0].get("client_order_id")
                elif existing_order_id:
                    conflicting_order_id = existing_order_id

                logger.warning(
                    f"Wash trade detected for {user_email}: {side} {qty} {symbol} "
                    f"conflicts with existing {opposite_side} order"
                )

                return {
                    "success": False,
                    "error": (
                        f"Cannot place {side} order: Alpaca detected a potential wash trade. "
                        f"You have a pending {opposite_side} order for {symbol}. "
                        "Please cancel it first or wait for it to fill."
                    ),
                    "error_code": "WASH_TRADE",
                    "conflicting_order_id": str(conflicting_order_id)
                    if conflicting_order_id
                    else None,
                    "alpaca_error": error_msg,
                }

            # For other errors, log and return
            logger.error(f"Failed to submit order for {user_email}: {submit_error}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to submit order: {error_msg}",
                "alpaca_error": error_msg,
            }

        # Store order in database
        order_doc = {
            "user_email": user_email,
            "alpaca_order_id": str(order.id) if order.id else None,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": order.status if hasattr(order, "status") else "pending",
            "filled_qty": float(order.filled_qty)
            if hasattr(order, "filled_qty") and order.filled_qty
            else 0,
            "filled_avg_price": float(order.filled_avg_price)
            if hasattr(order, "filled_avg_price") and order.filled_avg_price
            else None,
            "submitted_at": datetime.now(timezone.utc),
            "filled_at": None,
        }

        await db.user_orders.insert_one(order_doc)

        # If order is immediately filled, update positions
        if hasattr(order, "status") and order.status == "filled":
            await handle_order_fill(str(order.id), db)

        logger.info(
            f"Order submitted for {user_email}: {side} {qty} {symbol} (order_id: {order.id})"
        )

        return {
            "success": True,
            "order_id": str(order.id) if order.id else None,
            "client_order_id": client_order_id,
            "status": order.status if hasattr(order, "status") else "pending",
            "message": "Order submitted successfully",
        }

    except Exception as e:
        logger.error(f"Failed to execute trade for {user_email}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


async def handle_order_fill(alpaca_order_id: str, db) -> bool:
    """
    Process a filled order and update user balances/positions.

    Args:
        alpaca_order_id: Alpaca order ID
        db: Scoped database wrapper

    Returns:
        True if successful, False otherwise
    """
    try:
        # Get order from database - first find without user_email to get user_email
        # (since we don't know which user it belongs to at this point)
        # Note: This is the only place we query without user_email filter, but we immediately
        # extract user_email and use scoped helpers for all subsequent operations
        order = await db.user_orders.find_one({"alpaca_order_id": alpaca_order_id})
        if not order:
            logger.warning(f"Order not found in database: {alpaca_order_id}")
            return False

        user_email = order.get("user_email")
        if not user_email:
            logger.warning(f"Order missing user_email: {alpaca_order_id}")
            return False

        # Verify order exists for this user using scoped helper
        order = await get_user_order_by_alpaca_id(user_email, alpaca_order_id, db)
        if not order:
            logger.warning(f"Order not found for user: {alpaca_order_id}")
            return False

        # Get latest order status from Alpaca
        if not trading_client:
            return False

        try:
            alpaca_order = trading_client.get_order_by_id(alpaca_order_id)
        except Exception as e:
            logger.exception(f"Failed to get order from Alpaca: {e}")
            return False

        user_email = order.get("user_email")
        symbol = order.get("symbol")
        side = order.get("side")

        filled_qty = float(alpaca_order.filled_qty) if alpaca_order.filled_qty else 0
        filled_price = float(alpaca_order.filled_avg_price) if alpaca_order.filled_avg_price else 0

        if filled_qty == 0:
            return False  # Not filled yet

        # Update order status
        await db.user_orders.update_one(
            {"alpaca_order_id": alpaca_order_id},
            {
                "$set": {
                    "status": alpaca_order.status if hasattr(alpaca_order, "status") else "filled",
                    "filled_qty": filled_qty,
                    "filled_avg_price": filled_price,
                    "filled_at": datetime.now(timezone.utc),
                }
            },
        )

        # Update positions
        shares_to_add = filled_qty if side == "buy" else -filled_qty
        await add_user_position(user_email, symbol, shares_to_add, filled_price, db)

        # Update cash
        cost = filled_qty * filled_price
        cash_change = -cost if side == "buy" else cost
        await update_user_cash(user_email, cash_change, db)

        logger.info(f"Order filled: {user_email} {side} {filled_qty} {symbol} @ ${filled_price}")
        return True

    except Exception as e:
        logger.error(f"Failed to handle order fill: {e}", exc_info=True)
        return False


# ============================================================================
# ROUTES
# ============================================================================


@app.get("/login")
async def login_redirect(request: Request):
    """Redirect to auth hub login with redirect back to this app."""
    current_url = str(request.url)
    callback_url = f"{current_url.split('/login')[0]}/auth/callback"
    return RedirectResponse(
        url=f"{get_auth_hub_url()}/login?redirect_to={callback_url}", status_code=302
    )


@app.get("/auth/callback")
async def auth_callback(request: Request, token: str = None):
    """
    Token exchange endpoint - sets cookie for this app after auth hub login.
    Called with ?token=... after successful login at auth hub.
    """
    from urllib.parse import unquote_plus

    # Get token from query parameter (FastAPI auto-decodes, but handle URL-encoded tokens)
    if not token:
        token = request.query_params.get("token")

    if token:
        # URL-decode token in case it was encoded
        token = unquote_plus(token)

    if not token:
        return RedirectResponse(url=f"{get_auth_hub_url()}/login", status_code=302)

    # Validate token by getting user pool
    from mdb_engine.auth.shared_users import SharedUserPool

    pool: SharedUserPool = getattr(app.state, "user_pool", None)

    if not pool:
        return RedirectResponse(
            url=f"{get_auth_hub_url()}/login?error=pool_not_initialized", status_code=302
        )

    # Validate token
    user = await pool.validate_token(token)
    if not user:
        return RedirectResponse(
            url=f"{get_auth_hub_url()}/login?error=invalid_token", status_code=302
        )

    # Set cookie for this app
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="mdb_auth_token",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=86400,  # 24 hours
        path="/",
    )

    return response


@app.post("/logout")
async def logout(request: Request):
    """Logout and revoke token."""
    from mdb_engine.auth.shared_users import SharedUserPool

    pool: SharedUserPool = getattr(app.state, "user_pool", None)

    # Get token from cookie
    token = request.cookies.get("mdb_auth_token")

    # Revoke token if we have pool and token
    if pool and token:
        try:
            await pool.revoke_token(token, reason="logout")
        except (ValueError, RuntimeError, ConnectionError, OSError, TimeoutError) as e:
            logger.warning(f"Failed to revoke token: {e}", exc_info=True)

    # Create response redirecting to auth hub
    response = RedirectResponse(url=f"{get_auth_hub_url()}/login", status_code=302)

    # Delete cookie
    response.delete_cookie(
        "mdb_auth_token",
        path="/",
        domain=None,
        secure=False,
        samesite="lax",
    )

    return response


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db=Depends(get_scoped_db)):
    """Main page - shows Alpaca paper trading account registration/display."""
    # For shared auth, middleware sets request.state.user - check it directly
    user = get_current_user(request)
    if not user:
        # Redirect to auth hub if not authenticated
        return RedirectResponse(url=f"{get_auth_hub_url()}/login", status_code=302)

    user_email = user.get("email")
    roles = get_roles(user)

    # Check if user has a virtual account
    balance = await get_user_balance(user_email, db)
    has_account = balance is not None

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user": user,
            "roles": roles,
            "has_account": has_account,
            "app_name": "SSO App 2",
            "app_description": "Alpaca Paper Trading",
            "auth_hub_url": get_auth_hub_url(),
        },
    )


@app.get("/api/admin/stats")
async def get_admin_stats(request: Request, db=Depends(get_scoped_db)):
    """Get admin statistics - requires admin role."""
    # For shared auth, middleware sets request.state.user - check it directly
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if not is_admin(user):
        raise HTTPException(403, "Admin role required")

    # Get stats from all collections
    items_count = await db.items.count_documents({})
    users_count = (
        await db.users.count_documents({}) if "users" in await db.list_collection_names() else 0
    )

    return JSONResponse(
        {
            "success": True,
            "stats": {
                "items_count": items_count,
                "users_count": users_count,
                "app": APP_SLUG,
            },
            "admin": user.get("email"),
        }
    )


@app.delete("/api/admin/data/{item_id}")
async def delete_data(item_id: str, request: Request, db=Depends(get_scoped_db)):
    """Delete data - requires admin role."""
    # For shared auth, middleware sets request.state.user - check it directly
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if not is_admin(user):
        raise HTTPException(403, "Admin role required")

    from bson.objectid import ObjectId

    result = await db.items.delete_one({"_id": ObjectId(item_id)})

    if result.deleted_count == 0:
        raise HTTPException(404, "Item not found")

    return JSONResponse(
        {
            "success": True,
            "message": "Item deleted successfully",
        }
    )


@app.get("/api/me")
async def get_me(request: Request):
    """Get current user info."""
    # For shared auth, middleware sets request.state.user - check it directly
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    return {
        "email": user["email"],
        "roles": get_roles(user),
        "is_admin": is_admin(user),
        "app": APP_SLUG,
    }


# ============================================================================
# ALPACA API ROUTES
# ============================================================================


@app.post("/api/alpaca/register")
async def register_alpaca_account(request: Request, db=Depends(get_scoped_db)):
    """Register/create a virtual paper trading account for the current user (instant)."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_email = user.get("email")

    # Check if account already exists
    existing_balance = await get_user_balance(user_email, db)
    if existing_balance:
        return JSONResponse(
            {
                "success": False,
                "error": "Account already exists",
            },
            status_code=400,
        )

    # Create virtual account instantly
    balance_data = await initialize_user_balance(user_email, initial_cash=1000.0, db=db)
    if not balance_data:
        return JSONResponse(
            {
                "success": False,
                "error": "Failed to create virtual account. Please try again later.",
            },
            status_code=500,
        )

    # Get portfolio summary
    portfolio = await get_user_portfolio_value(user_email, db)

    return JSONResponse(
        {
            "success": True,
            "account": {
                "virtual_cash": balance_data.get("virtual_cash", 0),
                "buying_power": balance_data.get("buying_power", 0),
                "equity": balance_data.get("equity", 0),
                "status": "active",
                "currency": "USD",
            },
            "portfolio": portfolio,
            "message": "Virtual paper trading account created successfully",
        }
    )


@app.get("/api/alpaca/account")
async def get_alpaca_account(request: Request, db=Depends(get_scoped_db)):
    """Get virtual account details for the current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_email = user.get("email")

    # Get virtual balance from MongoDB
    balance = await get_user_balance(user_email, db)
    if not balance:
        return JSONResponse(
            {
                "success": False,
                "error": "No virtual account found. Please register first.",
            },
            status_code=404,
        )

    # Get portfolio summary
    portfolio = await get_user_portfolio_value(user_email, db)

    # Build account details
    account_details = {
        "virtual_cash": balance.get("virtual_cash", 0),
        "buying_power": balance.get("buying_power", 0),
        "equity": balance.get("equity", 0),
        "portfolio_value": portfolio.get("equity", 0),
        "status": "active",
        "currency": "USD",
        "positions_value": portfolio.get("positions_value", 0),
        "position_count": portfolio.get("position_count", 0),
    }

    return JSONResponse(
        {
            "success": True,
            "account": account_details,
            "created_at": balance.get("created_at").isoformat()
            if balance.get("created_at")
            else None,
        }
    )


@app.post("/api/alpaca/fund")
async def fund_alpaca_account(request: Request, db=Depends(get_scoped_db)):
    """Fund the current user's virtual account with $1k paper cash (instant)."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_email = user.get("email")
    balance = await get_user_balance(user_email, db)

    if not balance:
        raise HTTPException(404, "Account not found. Please register first.")

    # Update virtual cash instantly
    success = await update_user_cash(user_email, 1000.0, db)

    if success:
        return {"success": True, "message": "Account funded with $1,000"}
    else:
        return JSONResponse(
            {"success": False, "error": "Failed to fund account. Please try again."},
            status_code=500,
        )


@app.post("/api/alpaca/trade")
async def execute_trade_endpoint(request: Request, db=Depends(get_scoped_db)):
    """Execute a buy/sell order for the current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if not trading_client:
        raise HTTPException(
            503,
            "Alpaca service not configured. "
            "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.",
        )

    user_email = user.get("email")

    # Parse request body
    try:
        body = await request.json()
        symbol = body.get("symbol", "").upper().strip()
        side = body.get("side", "").lower().strip()
        qty = float(body.get("qty", 0))
        order_type = body.get("order_type", "market").lower().strip()
        limit_price = body.get("limit_price")

        if limit_price:
            limit_price = float(limit_price)
    except (ValueError, TypeError, KeyError) as e:
        return JSONResponse(
            {"success": False, "error": f"Invalid request body: {str(e)}"}, status_code=400
        )

    # Validate inputs
    if not symbol:
        return JSONResponse({"success": False, "error": "Symbol is required"}, status_code=400)

    if qty <= 0:
        return JSONResponse(
            {"success": False, "error": "Quantity must be greater than 0"}, status_code=400
        )

    if side not in ["buy", "sell"]:
        return JSONResponse(
            {"success": False, "error": "Side must be 'buy' or 'sell'"}, status_code=400
        )

    if order_type not in ["market", "limit"]:
        return JSONResponse(
            {"success": False, "error": "Order type must be 'market' or 'limit'"}, status_code=400
        )

    # Execute trade
    result = await execute_trade(
        user_email=user_email,
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        db=db,
    )

    # Clean result to ensure JSON serialization (convert datetime objects)
    cleaned_result = clean_mongo_doc(result) if result else {}

    if cleaned_result.get("success"):
        return JSONResponse(cleaned_result)
    else:
        return JSONResponse(cleaned_result, status_code=400)


@app.get("/api/alpaca/positions")
async def get_positions(request: Request, db=Depends(get_scoped_db)):
    """Get current positions for the current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_email = user.get("email")
    positions = await get_user_positions(user_email, db)

    # Convert MongoDB documents to JSON-serializable format
    cleaned_positions = clean_mongo_docs(positions)

    return JSONResponse({"success": True, "positions": cleaned_positions})


@app.get("/api/alpaca/orders")
async def get_orders(request: Request, db=Depends(get_scoped_db)):
    """Get order history for the current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_email = user.get("email")

    # Refresh pending orders from Alpaca before fetching
    if trading_client:
        await refresh_user_pending_orders(user_email, db)

    # Get orders from database using helper
    orders = await get_user_orders(user_email, db, limit=100)

    # Convert MongoDB documents to JSON-serializable format
    cleaned_orders = clean_mongo_docs(orders)

    return JSONResponse({"success": True, "orders": cleaned_orders})


@app.post("/api/alpaca/orders/{order_id}/cancel")
async def cancel_order(order_id: str, request: Request, db=Depends(get_scoped_db)):
    """Cancel a pending order."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    if not trading_client:
        raise HTTPException(503, "Alpaca service not configured")

    user_email = user.get("email")

    # Get order from database using helper
    order = await get_user_order_by_id(user_email, order_id, db)

    if not order:
        return JSONResponse({"success": False, "error": "Order not found"}, status_code=404)

    # Check if order can be cancelled
    status = order.get("status", "").lower()
    if status in ["filled", "cancelled", "expired", "rejected"]:
        return JSONResponse(
            {"success": False, "error": f"Cannot cancel order with status: {status}"},
            status_code=400,
        )

    # If already pending cancel, just return success
    if status == "pending_cancel":
        return JSONResponse({"success": True, "message": "Order cancellation already in progress"})

    alpaca_order_id = order.get("alpaca_order_id")
    if not alpaca_order_id:
        return JSONResponse(
            {"success": False, "error": "Order does not have Alpaca order ID"}, status_code=400
        )

    try:
        # Cancel order in Alpaca
        trading_client.cancel_order_by_id(alpaca_order_id)

        # Update order status in database (may be pending_cancel initially)
        await db.user_orders.update_one(
            {"alpaca_order_id": alpaca_order_id},
            {
                "$set": {
                    # Will be updated to cancelled when status refreshes
                    "status": "pending_cancel",
                    "cancelled_at": datetime.now(timezone.utc),
                }
            },
        )

        logger.info(f"Order cancellation requested: {alpaca_order_id} for {user_email}")

        return JSONResponse(
            {
                "success": True,
                "message": "Order cancellation requested. Status will update shortly.",
            }
        )

    except Exception as e:
        error_msg = str(e).lower()
        # Check if order was already cancelled or doesn't exist
        if "not found" in error_msg or "already cancelled" in error_msg or "404" in error_msg:
            # Update database to reflect cancelled status
            await db.user_orders.update_one(
                {"alpaca_order_id": alpaca_order_id},
                {"$set": {"status": "cancelled", "cancelled_at": datetime.now(timezone.utc)}},
            )
            return JSONResponse(
                {"success": True, "message": "Order was already cancelled or not found"}
            )

        logger.error(f"Failed to cancel order {alpaca_order_id}: {e}", exc_info=True)
        return JSONResponse(
            {"success": False, "error": f"Failed to cancel order: {str(e)}"}, status_code=500
        )


@app.get("/api/alpaca/portfolio")
async def get_portfolio(request: Request, db=Depends(get_scoped_db)):
    """Get complete portfolio summary for the current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")

    user_email = user.get("email")

    balance = await get_user_balance(user_email, db)
    if not balance:
        return JSONResponse(
            {
                "success": False,
                "error": "No account found. Please register first.",
            },
            status_code=404,
        )

    portfolio = await get_user_portfolio_value(user_email, db)
    positions = await get_user_positions(user_email, db)

    # Convert MongoDB documents to JSON-serializable format
    cleaned_positions = clean_mongo_docs(positions)

    return JSONResponse(
        {
            "success": True,
            "balance": {
                "virtual_cash": balance.get("virtual_cash", 0),
                "buying_power": balance.get("buying_power", 0),
                "equity": balance.get("equity", 0),
            },
            "portfolio": portfolio,
            "positions": cleaned_positions,
        }
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "app": APP_SLUG, "auth": "shared"}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting SSO App 2 on 0.0.0.0:8000")
    try:
        uvicorn.run(
            app, host="0.0.0.0", port=8000, log_level=os.getenv("LOG_LEVEL", "info").lower()
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)
