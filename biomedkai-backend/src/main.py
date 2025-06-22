import asyncio
import uvicorn
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
from typing import Optional, Dict, Any

from config.settings import settings, agent_config, load_agent_config
from src.api.websocket_server import websocket_router
from src.api.rest_api import api_router
from src.core.orchestrator import MedicalAgentOrchestrator
from src.models.llama_model import LlamaModelWrapper
from src.memory.hybrid_memory import HybridMemorySystem
from src.tools import create_tool_registry
from src.monitoring.metrics_collector import setup_metrics
from src.monitoring.health_checker import health_router
import os

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# Global instances
orchestrator: Optional[MedicalAgentOrchestrator] = None
memory_system: Optional[HybridMemorySystem] = None
tool_registry: Optional[Dict[str, Any]] = None
model_wrapper: Optional[LlamaModelWrapper] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global orchestrator, memory_system, tool_registry
    
    logger.info("Starting Medical AI Agent System", 
                app_name=settings.app_name,
                environment=settings.app_env)
    
    try:
        # Initialize model
        logger.info("Loading LLM model", model_path=settings.model_path)
        model_wrapper = LlamaModelWrapper(settings.model_path)
        model = await model_wrapper.initialize()
        
        # Initialize memory system
        logger.info("Initializing memory system")
        memory_system = HybridMemorySystem(
            redis_url=f"redis://{settings.redis_host}:{settings.redis_port}",
            postgres_url=f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
            neo4j_url=settings.neo4j_uri,
            neo4j_auth=(settings.neo4j_user, settings.neo4j_password)
        )
        await memory_system.initialize()
        
        # Create tool registry
        logger.info("Creating tool registry")
        tool_registry = create_tool_registry()

        model.set_context_retreiver(tool_registry.get("knowledge_graph_search"))
        
        # Initialize orchestrator
        logger.info("Initializing agent orchestrator")
        orchestrator = MedicalAgentOrchestrator(
            model=model,
            tools=tool_registry,
            memory_system=memory_system,
            config=agent_config
        )
        
        
        
        logger.info("Medical AI Agent System started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start Medical AI Agent System", error=str(e))
        raise
        
    finally:
        # Cleanup
        logger.info("Shutting down Medical AI Agent System")
        
        if memory_system:
            await memory_system.close()
            
        if hasattr(model_wrapper, 'cleanup'):
            await model_wrapper.cleanup()
            
        logger.info("Shutdown complete")


class EnhancedMediaServer:
    """
    Enhanced version of your MediaServer that integrates with the agent system
    """
    
    def __init__(self):
        # Your existing initialization
        self.model_name = "llama-3.1"
        self.connected_clients = set()
        
        # Initialize the agent system
        self._initialize_agent_system()
        
    def _initialize_agent_system(self):
        """Initialize the agent orchestrator"""
        # Initialize model wrapper
        self.model_wrapper = LlamaModelWrapper()
        asyncio.create_task(self.model_wrapper.initialize())
        
        # Initialize memory system
        self.memory_system = HybridMemorySystem(
            redis_url="redis://localhost:6379",
            postgres_url="postgresql://medical_ai:password@localhost:5432/medical_memory",
            neo4j_url="neo4j://localhost:7687",
            neo4j_auth=("neo4j", "password")
        )
        asyncio.create_task(self.memory_system.initialize())
        
        # Create tools
        self.tool_registry = create_tool_registry()
        
        # Load agent config
        agent_config = load_agent_config()
        
        self.model_wrapper.set_context_retreiver(self.tool_registry.get("knowledge_graph_search"))
        # Create orchestrator
        self.orchestrator = MedicalAgentOrchestrator(
            model=self.model_wrapper,
            tools=self.tool_registry,
            memory_system=self.memory_system,
            config=agent_config
        )
        
    async def handle_generate_response(self, data: Dict[str, Any], websocket):
        """Enhanced response generation using agent system"""
        prompt = data.get("prompt", "")
        chat_history = data.get("chat_history", [])
        use_agents = data.get("use_agents", True)
        
        await websocket.send(json.dumps({
            "type": "GENERATE_RESPONSE",
            "stream_start": True
        }))
        
        if use_agents:
            # Use the agent system
            async for chunk in self.orchestrator.process_query(prompt):
                await websocket.send(json.dumps({
                    "type": "GENERATE_RESPONSE",
                    "chunk": chunk,
                    "agent_mode": True
                }))
                await asyncio.sleep(0)
        else:
            # Use direct model (your existing code)
            async for chunk in self.model_wrapper.model.generate2(prompt, chat_history):
                await websocket.send(json.dumps({
                    "type": "GENERATE_RESPONSE",
                    "chunk": chunk
                }))
                await asyncio.sleep(0)
                
        await websocket.send(json.dumps({
            "type": "GENERATE_RESPONSE",
            "stream_end": True
        }))

# Create FastAPI app
app = FastAPI(
    title="Medical AI Agent System",
    description="Advanced medical AI assistant with multi-agent architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup monitoring
setup_metrics(app)

# Include routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(websocket_router, prefix="/ws")
app.include_router(health_router, prefix="/health")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Medical AI Agent System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "api": "/api/v1",
            "websocket": "/ws",
            "health": "/health",
            "docs": "/docs"
        }
    }


def main():
    """Main entry point"""
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()