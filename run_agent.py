#!/usr/bin/env python3
"""
Agent-only entry point for v3.
This script starts only the agent client without the server.
The agent expects the server to be running separately.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Load environment configuration from .env if present
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.client import run_multiprocess_client
from utils.env_helpers import get_env_int, get_env_str, parse_env_bool
from utils.langfuse_session import initialize_langfuse_session

logger = logging.getLogger(__name__)


def setup_logging(log_level: str | int) -> None:
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = log_level
    
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(level)


def main():
    """Main entry point for the Pokemon Agent client"""
    parser = argparse.ArgumentParser(description="Pokemon Emerald AI Agent Client")
    
    # Server connection
    parser.add_argument(
        "--server-host",
        type=str,
        default=get_env_str("AGENT_SERVER_HOST") or "localhost",
        help="Host where server is running (default: localhost or AGENT_SERVER_HOST env)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_int("AGENT_PORT", 8000),
        help="Port where server is running (default: 8000 or AGENT_PORT env)",
    )
    
    # Agent configuration
    available_backends = [
        "openai", 
        "openrouter", 
        "gemini", 
        "vllm", 
        "vertex",
        "azure_openai",
        "azure_openai_v1",
        "github_models"
    ]
    env_backend = os.getenv("AGENT_BACKEND") or os.getenv("POKEAGENT_BACKEND") or "gemini"
    env_backend = env_backend.lower()
    if env_backend not in available_backends:
        env_backend = "gemini"

    parser.add_argument(
        "--backend",
        type=str,
        choices=available_backends,
        default=env_backend,
        help="VLM backend (openai, openrouter, gemini, vllm, vertex)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("AGENT_MODEL_NAME"),
        help="Model name to use",
    )
    env_scaffold = (os.getenv("AGENT_SCAFFOLD") or "simple").lower()
    parser.add_argument(
        "--scaffold",
        type=str,
        default=env_scaffold,
        help="Agent scaffold: simple (default) or react (AGENT_SCAFFOLD env) or explorer",
    )
    parser.add_argument("--simple", action="store_true", 
                       help="DEPRECATED: Use --scaffold simple instead")
    
    # Operation modes
    parser.add_argument("--headless", action="store_true", 
                       help="Run without pygame display (headless)")
    parser.add_argument("--agent-auto", action="store_true", 
                       help="Agent acts automatically")
    parser.add_argument("--manual", action="store_true", 
                       help="Start in manual mode instead of agent mode")
    
    # Features
    parser.add_argument("--no-ocr", action="store_true",
                       help="Disable OCR dialogue detection (passed to agent logic)")

    # Logging configuration
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level (default: INFO)")

    args = parser.parse_args()
    args.backend = args.backend.lower()

    setup_logging(args.log_level)

    def apply_flag(attr: str, env_name: str):
        env_value = parse_env_bool(env_name)
        if env_value is None:
            return
        current = getattr(args, attr)
        if current and not env_value:
            # Command-line flag takes precedence if explicitly enabled
            return
        setattr(args, attr, env_value)

    apply_flag("agent_auto", "AGENT_AUTO")
    apply_flag("manual", "AGENT_MANUAL")
    apply_flag("headless", "AGENT_HEADLESS")
    apply_flag("no_ocr", "AGENT_NO_OCR")

    if not args.model_name:
        backend_fallback_models = {
            "openai": "gpt-5",
            "openrouter": "anthropic/claude-4.5-sonnet",
            "gemini": "gemini-2.5-flash",
            "vllm": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "vertex": "gemini-2.5-flash",
        }
        args.model_name = backend_fallback_models.get(args.backend)

    if args.backend == "vllm" and not args.model_name:
        parser.error("vLLM backend requires a model name. Set VLLM_MODEL in your .env or pass --model-name.")

    if not args.model_name:
        args.model_name = "gemini-2.5-flash"
    
    # Handle deprecated --simple flag
    if args.simple:
        print("⚠️ --simple is deprecated. Using --scaffold simple")
        args.scaffold = "simple"
    
    print("=" * 60)
    print("🤖 Pokemon Emerald AI Agent Client")
    print("=" * 60)
    
    # Display configuration
    print("\n🤖 Agent Configuration:")
    
    # Show agent mode and type
    agent_mode = os.getenv("AGENT_MODE", "legacy").lower()
    agent_type = os.getenv("AGENT_TYPE")
    
    print(f"   Agent Mode: {agent_mode}")
    
    if agent_mode == "custom":
        agent_type = agent_type or "minimal_vlm"
        print(f"   Agent Type: {agent_type}")
    else:
        # Legacy mode
        agent_type = agent_type or os.getenv("AGENT_SCAFFOLD") or args.scaffold
        scaffold_descriptions = {
            "simple": "Simple mode (direct frame→action)",
            "react": "ReAct agent (Thought→Action→Observation loop)"
        }
        description = scaffold_descriptions.get(agent_type, agent_type)
        print(f"   Agent Type: {description}")
    
    print(f"   Backend: {args.backend}")
    print(f"   Model: {args.model_name}")
    if args.no_ocr:
        print("   OCR: Disabled")
    
    print(f"\n📡 Connecting to server at: http://{args.server_host}:{args.port}")
    print(f"🎥 Stream View: http://{args.server_host}:{args.port}/stream")
    
    # Initialize Langfuse session ID with agent configuration
    overrides = {
        "AGENT_MODE": agent_mode,
        "AGENT_TYPE": agent_type,
        "AGENT_MODEL_NAME": args.model_name,
        "AGENT_BACKEND": args.backend,
    }
    session_id = initialize_langfuse_session(overrides)
    print(f"\n📊 Langfuse Session ID: {session_id}")
    
    print("\n🚀 Starting agent client...")
    print("-" * 60)

    try:
        # Run the client - it will connect to the external server
        success = run_multiprocess_client(server_port=args.port, args=args, server_host=args.server_host)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
        return 0
    
    finally:
        print("👋 Goodbye!")


if __name__ == "__main__":
    sys.exit(main())
