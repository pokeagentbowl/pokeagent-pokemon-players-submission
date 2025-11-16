#!/usr/bin/env python3
"""
Server-only entry point for v3.
This script starts only the server without the client.
"""

import sys
import os
import argparse

from dotenv import load_dotenv

# Load environment configuration from .env if present
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.env_helpers import get_env_int, get_env_str, parse_env_bool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokemon Emerald Server")
    
    # Server-specific arguments
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_int("AGENT_PORT", 8000),
        help="Port for web interface (default: 8000 or AGENT_PORT env)",
    )
    parser.add_argument(
        "--rom",
        type=str,
        default=get_env_str("AGENT_ROM") or "Emerald-GBAdvance/rom.gba",
        help="Path to ROM file (default: Emerald-GBAdvance/rom.gba or AGENT_ROM env)",
    )
    
    # State loading
    parser.add_argument(
        "--load-state",
        type=str,
        default=get_env_str("AGENT_LOAD_STATE"),
        help="Load a saved state file on startup (AGENT_LOAD_STATE env)",
    )
    parser.add_argument(
        "--load-checkpoint",
        action="store_true",
        help="Load from checkpoint files",
    )
    
    # Features
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record video of the gameplay",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR dialogue detection",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Enable manual mode with keyboard input",
    )
    
    args = parser.parse_args()
    
    if args.load_state:
        args.load_state = args.load_state.strip() or None
    
    def apply_flag(attr: str, env_name: str):
        env_value = parse_env_bool(env_name)
        if env_value is None:
            return
        current = getattr(args, attr)
        if current and not env_value:
            # Command-line flag takes precedence if explicitly enabled
            return
        setattr(args, attr, env_value)
    
    apply_flag("record", "AGENT_RECORD")
    apply_flag("no_ocr", "AGENT_NO_OCR")
    apply_flag("manual", "AGENT_MANUAL")
    
    if args.load_checkpoint:
        # Auto-load checkpoint.state when --load-checkpoint is used
        checkpoint_state = ".pokeagent_cache/checkpoint.state"
        if os.path.exists(checkpoint_state):
            args.load_state = checkpoint_state
            # Set environment variable to enable LLM checkpoint loading
            os.environ["LOAD_CHECKPOINT_MODE"] = "true"
            print(f"🔄 Server will load checkpoint: {checkpoint_state}")
            print(f"🔄 LLM metrics will be restored from .pokeagent_cache/checkpoint_llm.txt")
        else:
            print(f"⚠️ Checkpoint file not found: {checkpoint_state}")
    
    # Set environment variable for load state if specified
    if args.load_state:
        os.environ["LOAD_STATE"] = args.load_state
    
    print("=" * 60)
    print("📡 Pokemon Emerald Server")
    print("=" * 60)
    print(f"\n🎮 Configuration:")
    print(f"   Port: {args.port}")
    print(f"   ROM: {args.rom}")
    if args.load_state:
        print(f"   Load State: {args.load_state}")
    if args.record:
        print("   Recording: Enabled")
    if args.no_ocr:
        print("   OCR: Disabled")
    if args.manual:
        print("   Mode: Manual")
    print(f"\n🎥 Stream View: http://127.0.0.1:{args.port}/stream")
    print("\n🚀 Starting server...")
    print("-" * 60)
    
    # Import and run the server main function
    from server.app import main
    main()
