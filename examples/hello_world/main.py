#!/usr/bin/env python3
"""
Hello World Example for MDB_RUNTIME

This example demonstrates the basic usage of MDB_RUNTIME with a single-app setup.
It shows how to initialize the engine, register an app, and perform basic CRUD operations.
"""
import asyncio
import os
from pathlib import Path
from datetime import datetime

from mdb_runtime import RuntimeEngine


async def main():
    """Main example function"""
    print("🚀 Initializing MDB_RUNTIME...")
    
    # Get MongoDB connection from environment
    # In Docker Compose, this is set automatically
    mongo_uri = os.getenv(
        "MONGO_URI", 
        "mongodb://admin:password@mongodb:27017/?authSource=admin"
    )
    db_name = os.getenv("MONGO_DB_NAME", "hello_world_db")
    
    # Mask password in logs
    safe_uri = mongo_uri.replace("password", "***") if "password" in mongo_uri else mongo_uri
    print(f"📡 Connecting to MongoDB: {safe_uri}")
    print(f"📦 Database: {db_name}\n")
    
    # Initialize the runtime engine
    engine = RuntimeEngine(
        mongo_uri=mongo_uri,
        db_name=db_name
    )
    
    try:
        # Connect to MongoDB
        await engine.initialize()
        print("✅ Engine initialized successfully")
        
        # Load and register the app manifest
        # Works in both Docker (/app) and local development
        manifest_path = Path("/app/manifest.json")
        if not manifest_path.exists():
            # Fallback for local development
            manifest_path = Path(__file__).parent / "manifest.json"
            if not manifest_path.exists():
                print(f"❌ Manifest not found. Tried: /app/manifest.json and {manifest_path}")
                return
        
        manifest = await engine.load_manifest(manifest_path)
        success = await engine.register_app(manifest, create_indexes=True)
        
        if not success:
            print("❌ Failed to register app")
            return
        
        print(f"✅ App '{manifest['slug']}' registered successfully\n")
        
        # Get a scoped database for the app
        # All operations will be automatically scoped to "hello_world"
        db = engine.get_scoped_db("hello_world")
        
        # ============================================
        # CREATE: Insert some sample data
        # ============================================
        print("📝 Creating sample data...")
        
        greetings = [
            {"message": "Hello, World!", "language": "en", "created_at": datetime.utcnow()},
            {"message": "Hello, MDB_RUNTIME!", "language": "en", "created_at": datetime.utcnow()},
            {"message": "Hello, Python!", "language": "en", "created_at": datetime.utcnow()},
        ]
        
        for greeting in greetings:
            result = await db.greetings.insert_one(greeting)
            print(f"✅ Created greeting: {greeting['message']}")
        
        print()
        
        # ============================================
        # READ: Query the data
        # ============================================
        print("🔍 Querying data...")
        
        # Find all greetings
        # Note: This automatically filters by app_id - you don't need to specify it!
        all_greetings = await db.greetings.find({}).to_list(length=10)
        print(f"✅ Found {len(all_greetings)} greetings")
        
        for greeting in all_greetings:
            print(f"   - {greeting['message']} (language: {greeting['language']})")
        
        print()
        
        # Find greetings by language
        # The app_id filter is still automatically applied
        english_greetings = await db.greetings.find({"language": "en"}).to_list(length=10)
        print(f"🔍 Finding English greetings...")
        print(f"✅ Found {len(english_greetings)} English greetings")
        print()
        
        # ============================================
        # UPDATE: Modify existing data
        # ============================================
        print("✏️  Updating greeting...")
        
        # Update the first greeting
        if all_greetings:
            greeting_id = all_greetings[0]["_id"]
            update_result = await db.greetings.update_one(
                {"_id": greeting_id},
                {"$set": {"message": "Hello, Updated World!"}}
            )
            
            if update_result.modified_count > 0:
                print("✅ Updated greeting: Hello, Updated World!")
            else:
                print("⚠️  No greeting was updated")
        
        print()
        
        # ============================================
        # READ: Verify the update
        # ============================================
        print("🔍 Verifying update...")
        
        updated_greeting = await db.greetings.find_one({"message": "Hello, Updated World!"})
        if updated_greeting:
            print(f"✅ Found updated greeting: {updated_greeting['message']}")
        else:
            print("⚠️  Updated greeting not found")
        
        print()
        
        # ============================================
        # Health Check
        # ============================================
        print("📊 Health Status:")
        health = await engine.get_health_status()
        print(f"   - Status: {health['status']}")
        
        # Extract engine health details
        engine_check = next((c for c in health['checks'] if c.get('name') == 'engine'), None)
        if engine_check and engine_check.get('details'):
            details = engine_check['details']
            print(f"   - App Count: {details.get('app_count', 0)}")
            print(f"   - Initialized: {details.get('initialized', False)}")
        else:
            print(f"   - App Count: 0")
            print(f"   - Initialized: False")
        print()
        
        # ============================================
        # Understanding What Happened
        # ============================================
        print("💡 What happened behind the scenes:")
        print("   - All documents were automatically tagged with app_id='hello_world'")
        print("   - All queries were automatically filtered by app_id")
        print("   - Indexes were created automatically from the manifest")
        print("   - You never had to think about app isolation - it just works!")
        print()
        
        print("✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await engine.shutdown()
        print("\n🧹 Cleaned up and shut down")


if __name__ == "__main__":
    asyncio.run(main())

