"""
Single chat endpoint with authentication and conversation memory
"""

import logging

from fastapi import APIRouter, HTTPException, Request

import src.config as config
from src.agents.langgraph_workflow import run_workflow
from src.config import (
    EMBED_DIMENSIONS,
    get_client_by_provider,
    get_supabase,
)
from src.models.chat import ChatRequest, ChatResponse, IntentClassificationResponse
from src.utils.auth import AuthenticationError, authenticate_request
from src.utils.conversation_memory import generate_conversation_id, store_message, store_message_with_embedding

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request_body: ChatRequest, request: Request):
    """
    Main chat endpoint with authentication, LangGraph workflow, and conversation memory

    Requires Authorization header: Bearer <supabase-jwt-token>

    Process:
    1. Authenticate user by verifying JWT token
    2. Generate or use existing conversation_id
    3. Store user message with embedding
    4. Run LangGraph workflow:
       - RAG Retrieval: Search documents and conversation history
       - Intent Classification: RAG query vs context-based coaching
       - Context Classification: Classify into Store A context
         (Professional, Psychological, Learning, Social, Emotional, Aspirational)
       - Response Generation: Generate personalized response
    5. Store assistant response with embedding
    6. Return response with conversation_id, intent classification, context classification, and sources

    Args:
        request_body: Chat message and optional conversation_id
        request: FastAPI request object (to access headers)

    Returns:
        ChatResponse with generated response, conversation_id, sources, mood analysis,
        context classification, and intent classification

    Raises:
        HTTPException 401: If authentication fails
        HTTPException 500: If chat processing fails
    """
    # Step 1: Authenticate user
    try:
        auth_header = request.headers.get("Authorization")
        user_info = authenticate_request(auth_header)
        user_id = user_info["user_id"]
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    # Step 2: Process chat message with conversation memory
    try:
        # Use request parameters or fall back to config defaults
        embed_provider = request_body.embed_provider
        embed_model = request_body.embed_model
        chat_provider = request_body.chat_provider
        chat_model = request_body.chat_model

        # Get the appropriate embedding and chat clients based on provider
        embed_client = get_client_by_provider(embed_provider)
        chat_client = get_client_by_provider(chat_provider)

        # Generate conversation_id if not provided
        conversation_id = request_body.conversation_id or generate_conversation_id()

        # Run LangGraph workflow (includes RAG retrieval)
        workflow_state = await run_workflow(
            message=request_body.message,
            user_id=user_id,
            conversation_id=conversation_id,
            chat_client=chat_client,
            embed_client=embed_client,
            supabase=get_supabase(),
            embed_model=embed_model,
            embed_dimensions=EMBED_DIMENSIONS,
            chat_model=chat_model,
            intent_classifier_type=request_body.intent_classifier_type,
            semantic_gate_enabled=request_body.semantic_gate_enabled,
        )

        # Extract workflow results
        response_text = workflow_state["response"]
        unified_classification = workflow_state.get("unified_classification")
        sources = workflow_state["sources"]
        conversation_history = workflow_state["conversation_history"]

        # Append workflow process to response if verbose mode is enabled
        if config.VERBOSE_MODE:
            workflow_steps = workflow_state.get("workflow_process", [])
            if workflow_steps:
                # Markdown needs two trailing spaces + newline for line breaks
                workflow_text = "\n\n=========================================================="
                workflow_text += "\n[WORKFLOW START]"
                workflow_text += "\n\n" + "\n".join(step + "  " for step in workflow_steps)
                workflow_text += "\n[WORKFLOW END]"
                workflow_text += "\n\n==========================================================="
                response_text = response_text + workflow_text

        # Step 5: Store conversation (embedding only for worthy messages)
        # TODO: This is a placeholder for the worthy message filtering logic
        is_worthy = False

        if is_worthy:
            store_message_with_embedding(
                supabase=get_supabase(),
                embed_client=embed_client,
                embed_model=embed_model,
                conversation_id=conversation_id,
                user_id=user_id,
                role="user",
                message=request_body.message,
            )
            store_message_with_embedding(
                supabase=get_supabase(),
                embed_client=embed_client,
                embed_model=embed_model,
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                message=response_text,
            )
        else:
            store_message(
                supabase=get_supabase(),
                conversation_id=conversation_id,
                user_id=user_id,
                role="user",
                message=request_body.message,
            )
            store_message(
                supabase=get_supabase(),
                conversation_id=conversation_id,
                user_id=user_id,
                role="assistant",
                message=response_text,
            )

        # Prepare conversation context for response
        conversation_context_list = [
            {"role": msg["role"], "message": msg["message"][:100] + "...", "similarity": msg["similarity"]} for msg in conversation_history
        ]

        # Prepare intent classification for response
        classification_response = None
        if unified_classification:
            classification_response = IntentClassificationResponse(
                category=unified_classification.category.value,
                confidence=unified_classification.confidence,
                secondary_categories=[cat.value for cat in unified_classification.secondary_categories],
                reasoning=unified_classification.reasoning,
                key_entities=unified_classification.key_entities,
            )

        return ChatResponse(
            response=response_text,
            user_id=user_id,
            conversation_id=conversation_id,
            sources=sources,
            conversation_context=conversation_context_list,
            classification=classification_response,
        )

    except Exception as e:
        logger.exception("Chat processing failed")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")
