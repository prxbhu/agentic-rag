"""
System verification script to check all components
"""
import asyncio
import sys
from sqlalchemy import text, inspect
import httpx

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_status(message, status="info"):
    """Print colored status message"""
    if status == "success":
        print(f"{GREEN}✓{RESET} {message}")
    elif status == "error":
        print(f"{RED}✗{RESET} {message}")
    elif status == "warning":
        print(f"{YELLOW}⚠{RESET} {message}")
    else:
        print(f"{BLUE}ℹ{RESET} {message}")


async def verify_database():
    """Verify database connection and schema"""
    print("\n" + "="*50)
    print("Checking Database...")
    print("="*50)
    
    try:
        from app.database import engine
        from app.config import settings
        
        async with engine.begin() as conn:
            # Test connection
            await conn.execute(text("SELECT 1"))
            print_status("Database connection successful", "success")
            
            # Check pgvector extension
            result = await conn.execute(text(
                "SELECT * FROM pg_extension WHERE extname = 'vector'"
            ))
            if result.fetchone():
                print_status("pgvector extension enabled", "success")
            else:
                print_status("pgvector extension NOT enabled", "error")
                return False
            
            # Check tables
            def get_tables(sync_conn):
                inspector = inspect(sync_conn)
                return inspector.get_table_names()

            tables = await conn.run_sync(get_tables)
            
            required_tables = [
                'workspaces', 'resources', 'chunks', 'conversations',
                'messages', 'embedding_tasks', 'source_quality'
            ]
            
            for table in required_tables:
                if table in tables:
                    print_status(f"Table '{table}' exists", "success")
                else:
                    print_status(f"Table '{table}' MISSING", "error")
                    return False
            
            # Check indexes
            result = await conn.execute(text("""
                SELECT indexname FROM pg_indexes 
                WHERE schemaname = 'public' AND indexname LIKE 'idx_%'
            """))
            indexes = [row[0] for row in result.fetchall()]
            print_status(f"Found {len(indexes)} indexes", "success")
            
            # Check hybrid search function
            result = await conn.execute(text("""
                SELECT proname FROM pg_proc WHERE proname = 'hybrid_search'
            """))
            if result.fetchone():
                print_status("Hybrid search function exists", "success")
            else:
                print_status("Hybrid search function MISSING", "error")
                return False
            
            # Check default workspace
            result = await conn.execute(text("SELECT COUNT(*) FROM workspaces"))
            count = result.scalar()
            if count > 0:
                print_status(f"Found {count} workspace(s)", "success")
            else:
                print_status("No workspaces found", "warning")
            
            return True
            
    except Exception as e:
        print_status(f"Database check failed: {e}", "error")
        return False


def verify_redis():
    """Verify Redis connection"""
    print("\n" + "="*50)
    print("Checking Redis...")
    print("="*50)
    
    try:
        import redis
        from app.config import settings
        
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        print_status("Redis connection successful", "success")
        
        info = r.info()
        print_status(f"Redis version: {info.get('redis_version')}", "success")
        print_status(f"Used memory: {info.get('used_memory_human')}", "success")
        
        return True
    except Exception as e:
        print_status(f"Redis check failed: {e}", "error")
        return False


async def verify_ollama():
    """Verify Ollama connection"""
    print("\n" + "="*50)
    print("Checking Ollama...")
    print("="*50)
    
    try:
        from app.config import settings
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=10.0
            )
            
            if response.status_code == 200:
                print_status("Ollama connection successful", "success")
                
                data = response.json()
                models = data.get("models", [])
                
                if models:
                    print_status(f"Found {len(models)} model(s):", "success")
                    for model in models:
                        print(f"  - {model['name']}")
                    
                    # Check if required model is available
                    model_names = [m["name"] for m in models]
                    if any(settings.CHAT_MODEL in name for name in model_names):
                        print_status(f"Required model '{settings.CHAT_MODEL}' available", "success")
                    else:
                        print_status(f"Required model '{settings.CHAT_MODEL}' NOT available", "warning")
                        print(f"  Run: ollama pull {settings.CHAT_MODEL}")
                else:
                    print_status("No models found", "warning")
                    print(f"  Run: ollama pull {settings.CHAT_MODEL}")
                
                return True
            else:
                print_status(f"Ollama returned HTTP {response.status_code}", "error")
                return False
                
    except Exception as e:
        print_status(f"Ollama check failed: {e}", "error")
        return False


def verify_embedding_service():
    """Verify embedding service"""
    print("\n" + "="*50)
    print("Checking Embedding Service...")
    print("="*50)
    
    try:
        from app.services.embedding import embedding_service
        
        info = embedding_service.get_model_info()
        print_status(f"Model: {info['model_name']}", "success")
        print_status(f"Dimension: {info['dimension']}", "success")
        print_status(f"Device: {info['device']}", "success")
        
        # Test embedding generation
        test_embedding = embedding_service.embed_text("test")
        print_status(f"Generated test embedding ({len(test_embedding)} dimensions)", "success")
        
        return True
    except Exception as e:
        print_status(f"Embedding service check failed: {e}", "error")
        return False


def verify_hardware():
    """Verify hardware detection"""
    print("\n" + "="*50)
    print("Checking Hardware...")
    print("="*50)
    
    try:
        from app.services.hardware import HardwareDetector
        
        info = HardwareDetector.get_system_info()
        
        print_status(f"OS: {info['os']}", "success")
        print_status(f"CPU cores: {info['cpu_cores']}", "success")
        print_status(f"RAM: {info['ram_available_gb']}GB / {info['ram_total_gb']}GB", "success")
        print_status(f"GPU available: {info['has_gpu']}", 
                    "success" if info['has_gpu'] else "warning")
        
        if info['has_gpu']:
            print_status(f"GPU type: {info['gpu_type']}", "success")
        
        optimal_model = HardwareDetector.get_optimal_model()
        print_status(f"Optimal model: {optimal_model}", "success")
        
        return True
    except Exception as e:
        print_status(f"Hardware check failed: {e}", "error")
        return False


def verify_python_files():
    """Verify all required Python files exist"""
    print("\n" + "="*50)
    print("Checking Python Files...")
    print("="*50)
    
    required_files = [
        "app/__init__.py",
        "app/main.py",
        "app/config.py",
        "app/database.py",
        "app/agents/__init__.py",
        "app/agents/rag_agent.py",
        "app/agents/tools.py",
        "app/agents/prompts.py",
        "app/api/__init__.py",
        "app/api/resources.py",
        "app/api/conversations.py",
        "app/api/health.py",
        "app/services/__init__.py",
        "app/services/embedding.py",
        "app/services/search.py",
        "app/services/ranking.py",
        "app/services/citation.py",
        "app/services/hardware.py",
        "app/services/llm_service.py",
        "app/services/ingestion.py",
        "app/models/__init__.py",
        "app/models/schemas.py",
        "app/models/database_models.py",
        "app/tasks/__init__.py",
        "app/tasks/celery_app.py",
        "app/tasks/embedding_tasks.py",
    ]
    
    import os
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            # Check if file has content
            if os.path.getsize(file_path) > 0:
                print_status(f"{file_path}", "success")
            else:
                print_status(f"{file_path} (EMPTY)", "warning")
                all_exist = False
        else:
            print_status(f"{file_path} (MISSING)", "error")
            all_exist = False
    
    return all_exist


async def main():
    """Run all verifications"""
    print("\n" + "="*70)
    print(" "*15 + "RAG SYSTEM VERIFICATION")
    print("="*70)
    
    results = {
        "Python Files": verify_python_files(),
        "Hardware": verify_hardware(),
        "Database": await verify_database(),
        "Redis": verify_redis(),
        "Ollama": await verify_ollama(),
        "Embedding Service": verify_embedding_service()
    }
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for component, passed in results.items():
        status = "PASS" if passed else "FAIL"
        color = GREEN if passed else RED
        print(f"{component:.<40} {color}{status}{RESET}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print(f"\n{GREEN}✓ All checks passed! System is ready.{RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{RED}✗ Some checks failed. Please fix the issues above.{RESET}\n")
        print("Run: python3 init_db.py  # To initialize database")
        print("See SETUP.md for detailed troubleshooting")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())