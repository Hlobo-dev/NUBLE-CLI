#!/usr/bin/env python3
"""
KYPERIAN Elite - FastAPI Backend

Professional-grade API for the KYPERIAN multi-agent financial advisor.

Features:
- REST endpoints for chat and analysis
- WebSocket for real-time streaming
- Quick lookup endpoints for market data
- User session management
- Health checks and monitoring
"""

import os
import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any, List
import logging

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query, Path, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..agents import OrchestratorAgent, OrchestratorConfig
from ..memory import MemoryManager, UserProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ Pydantic Models ============

if HAS_FASTAPI:
    class ChatRequest(BaseModel):
        """Request model for chat endpoint."""
        message: str = Field(..., description="User's message")
        conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
        user_id: Optional[str] = Field(None, description="User ID")
        context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    class ChatResponse(BaseModel):
        """Response model for chat endpoint."""
        message: str
        data: Dict[str, Any] = {}
        charts: List[Dict] = []
        actions: List[Dict] = []
        confidence: float = 0.0
        agents_used: List[str] = []
        execution_time_seconds: float = 0.0
        conversation_id: str = ""
        symbols: List[str] = []
    
    class QuickLookupResponse(BaseModel):
        """Response for quick lookup endpoints."""
        symbol: str
        data: Dict[str, Any]
        timestamp: str
    
    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        version: str
        uptime_seconds: float
        agents_available: int
        memory_stats: Dict[str, Any]
    
    class UserProfileRequest(BaseModel):
        """Request for user profile."""
        user_id: str
        name: Optional[str] = None
        risk_tolerance: str = "moderate"
        portfolio: Dict[str, float] = {}
        watchlist: List[str] = []
        preferences: Dict[str, Any] = {}


# ============ App Factory ============

def create_app(
    api_key: str = None,
    enable_memory: bool = True,
    enable_cors: bool = True
) -> 'FastAPI':
    """
    Create the KYPERIAN FastAPI application.
    
    Args:
        api_key: Anthropic API key (defaults to env var)
        enable_memory: Enable persistent memory
        enable_cors: Enable CORS middleware
        
    Returns:
        Configured FastAPI application
    """
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")
    
    # Create app
    app = FastAPI(
        title="KYPERIAN Elite API",
        description="The world's most intelligent financial advisor - Multi-Agent Cognitive System",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
    
    # Initialize components
    app.state.start_time = datetime.now()
    app.state.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    
    # Orchestrator
    config = OrchestratorConfig(use_opus=True)
    app.state.orchestrator = OrchestratorAgent(
        api_key=app.state.api_key,
        config=config
    )
    
    # Memory
    if enable_memory:
        app.state.memory = MemoryManager()
    else:
        app.state.memory = None
    
    # WebSocket connections
    app.state.active_connections: Dict[str, WebSocket] = {}
    
    # ============ Routes ============
    
    @app.get("/", response_model=HealthResponse)
    async def root():
        """Root endpoint - health check."""
        return await health_check()
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        uptime = (datetime.now() - app.state.start_time).total_seconds()
        
        agents_available = len(app.state.orchestrator.get_available_agents())
        
        memory_stats = {}
        if app.state.memory:
            memory_stats = app.state.memory.get_stats()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime_seconds=uptime,
            agents_available=agents_available,
            memory_stats=memory_stats
        )
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Main chat endpoint.
        
        Send a message and get a comprehensive response from the multi-agent system.
        """
        try:
            # Generate IDs if not provided
            conversation_id = request.conversation_id or str(uuid.uuid4())
            user_id = request.user_id or "anonymous"
            
            # Get user context from memory
            user_context = {}
            if app.state.memory and user_id != "anonymous":
                profile = app.state.memory.get_user_profile(user_id)
                if profile:
                    user_context = profile.to_dict()
            
            # Merge with request context
            if request.context:
                user_context.update(request.context)
            
            # Process through orchestrator
            result = await app.state.orchestrator.process(
                user_message=request.message,
                conversation_id=conversation_id,
                user_context=user_context
            )
            
            return ChatResponse(
                message=result['message'],
                data=result.get('data', {}),
                charts=result.get('charts', []),
                actions=result.get('actions', []),
                confidence=result.get('confidence', 0),
                agents_used=result.get('agents_used', []),
                execution_time_seconds=result.get('execution_time_seconds', 0),
                conversation_id=conversation_id,
                symbols=result.get('symbols', [])
            )
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/chat/stream")
    async def chat_stream(request: ChatRequest):
        """
        Streaming chat endpoint.
        
        Returns a server-sent events stream for real-time responses.
        """
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        async def generate():
            try:
                # Send initial response
                yield f"data: {json.dumps({'type': 'start', 'conversation_id': conversation_id})}\n\n"
                
                # Process
                result = await app.state.orchestrator.process(
                    user_message=request.message,
                    conversation_id=conversation_id,
                    user_context=request.context or {}
                )
                
                # Stream agents used
                for agent in result.get('agents_used', []):
                    yield f"data: {json.dumps({'type': 'agent', 'agent': agent})}\n\n"
                    await asyncio.sleep(0.1)
                
                # Stream final response
                yield f"data: {json.dumps({'type': 'response', 'data': result})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    
    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket endpoint for real-time chat.
        
        Supports bidirectional communication for live updates.
        """
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        app.state.active_connections[connection_id] = websocket
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                user_message = message_data.get('message', '')
                conversation_id = message_data.get('conversation_id', connection_id)
                user_context = message_data.get('context', {})
                
                # Send acknowledgment
                await websocket.send_json({
                    'type': 'received',
                    'conversation_id': conversation_id
                })
                
                # Process
                try:
                    result = await app.state.orchestrator.process(
                        user_message=user_message,
                        conversation_id=conversation_id,
                        user_context=user_context
                    )
                    
                    await websocket.send_json({
                        'type': 'response',
                        'data': result
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        'type': 'error',
                        'message': str(e)
                    })
                    
        except WebSocketDisconnect:
            del app.state.active_connections[connection_id]
    
    @app.get("/quick/{symbol}", response_model=QuickLookupResponse)
    async def quick_lookup(symbol: str = Path(..., description="Stock/crypto symbol")):
        """
        Quick symbol lookup.
        
        Get basic information about a symbol without full analysis.
        """
        try:
            # Quick query to market analyst only
            result = await app.state.orchestrator.process(
                user_message=f"Quick price check for {symbol}",
                conversation_id=f"quick_{symbol}_{datetime.now().timestamp()}",
                user_context={}
            )
            
            return QuickLookupResponse(
                symbol=symbol.upper(),
                data=result.get('data', {}),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/agents")
    async def list_agents():
        """List all available agents."""
        return {
            'agents': app.state.orchestrator.get_available_agents(),
            'count': len(app.state.orchestrator.get_available_agents())
        }
    
    # ============ User Profile Routes ============
    
    @app.post("/users/profile")
    async def create_or_update_profile(request: UserProfileRequest):
        """Create or update a user profile."""
        if not app.state.memory:
            raise HTTPException(status_code=503, detail="Memory not enabled")
        
        profile = UserProfile(
            user_id=request.user_id,
            name=request.name,
            risk_tolerance=request.risk_tolerance,
            portfolio=request.portfolio,
            watchlist=request.watchlist,
            preferences=request.preferences
        )
        
        app.state.memory.save_user_profile(profile)
        
        return {'status': 'success', 'user_id': request.user_id}
    
    @app.get("/users/{user_id}/profile")
    async def get_profile(user_id: str):
        """Get a user profile."""
        if not app.state.memory:
            raise HTTPException(status_code=503, detail="Memory not enabled")
        
        profile = app.state.memory.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        return profile.to_dict()
    
    @app.get("/users/{user_id}/conversations")
    async def get_conversations(user_id: str, limit: int = Query(50, ge=1, le=100)):
        """Get user's conversation history."""
        if not app.state.memory:
            raise HTTPException(status_code=503, detail="Memory not enabled")
        
        conversations = app.state.memory.get_user_conversations(user_id, limit)
        
        return {
            'user_id': user_id,
            'conversations': [c.to_dict() for c in conversations],
            'count': len(conversations)
        }
    
    @app.get("/users/{user_id}/predictions")
    async def get_prediction_accuracy(user_id: str):
        """Get prediction accuracy for a user."""
        if not app.state.memory:
            raise HTTPException(status_code=503, detail="Memory not enabled")
        
        accuracy = app.state.memory.get_prediction_accuracy(user_id=user_id)
        
        return {
            'user_id': user_id,
            'accuracy': accuracy
        }
    
    @app.post("/feedback")
    async def submit_feedback(
        user_id: str,
        conversation_id: str,
        rating: int = Query(..., ge=1, le=5),
        comment: Optional[str] = None
    ):
        """Submit feedback for a conversation."""
        if not app.state.memory:
            raise HTTPException(status_code=503, detail="Memory not enabled")
        
        app.state.memory.save_feedback(user_id, conversation_id, rating, comment)
        
        return {'status': 'success'}
    
    return app


# Create default app instance
app = None
if HAS_FASTAPI:
    app = create_app()


# ============ CLI Runner ============

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    Run the KYPERIAN API server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn not installed. Run: pip install uvicorn")
    
    logger.info(f"Starting KYPERIAN Elite API on {host}:{port}")
    
    uvicorn.run(
        "kyperian.api.main:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    run_server()
