#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Copyright (c) 2025 Prabin Kumar Rath
# Co-developed with Claude Sonnet 4.5 (Anthropic)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

"""
Policy Server for Franka FR3 Deployment

This script starts a policy server that loads your trained policy 
(ACT, Diffusion, VQ-BeT, SmolVLA, GROOT, PI0, PI05, etc.) 
and serves inference requests from Franka robot clients asynchronously.

Usage:
    python deploy_policy_server.py [--host HOST] [--port PORT] [--fps FPS]

Example:
    python deploy_policy_server.py --host 127.0.0.1 --port 8080 --fps 10
"""

import argparse
import logging

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Start policy server for Franka FR3 deployment (supports ACT, Diffusion, VQ-BeT, SmolVLA, GROOT, PI0, PI05)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Server configuration
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host address to bind the server to (use 0.0.0.0 for all interfaces)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080,
        help="Port number to bind the server to"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=10,
        help="Control frequency in frames per second (should match training data frequency)"
    )
    parser.add_argument(
        "--similarity_atol",
        type=float,
        default=1.0,
        help="Joint-space L2 tolerance for skipping 'similar' observations (lower = more observations permitted)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not 1 <= args.port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {args.port}")
    
    if args.fps <= 0:
        raise ValueError(f"FPS must be positive, got {args.fps}")
    
    # Create policy server configuration
    config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        similarity_atol=args.similarity_atol,
    )
    
    logger.info("="*60)
    logger.info("Franka FR3 Policy Server")
    logger.info("="*60)
    logger.info(f"Host: {config.host}")
    logger.info(f"Port: {config.port}")
    logger.info(f"FPS: {config.fps}")
    logger.info(f"Environment dt: {config.environment_dt:.4f}s")
    logger.info(f"Similarity atol: {config.similarity_atol}")
    logger.info("="*60)
    logger.info("Server is ready. Waiting for Franka robot client connection...")
    logger.info("The policy will be loaded when the first client connects.")
    logger.info("Supported policies: ACT, Diffusion, VQ-BeT, SmolVLA, GROOT, PI0, PI05")
    logger.info("Press Ctrl+C to stop the server.")
    logger.info("="*60)
    
    try:
        # Start the policy server
        serve(config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    main()
    