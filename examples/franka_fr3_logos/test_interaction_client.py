"""Test client for interaction_server.py WebSocket commands.

Sends task, start, and stop commands one at a time when Enter is pressed.
"""

import asyncio
import json
import websockets

SERVER_URL = "ws://localhost:8765"


async def send_command(ws, command: dict):
    """Send a command and print the response."""
    await ws.send(json.dumps(command))
    response = await ws.recv()
    print(f"Sent: {command}")
    print(f"Response: {response}")
    print("-" * 50)
    return json.loads(response)


def wait_for_enter(prompt: str):
    """Wait for user to press Enter."""
    input(prompt)


async def main():
    async with websockets.connect(SERVER_URL, ping_timeout=None) as ws:
        # Task command template
        task_cmd = {
            "command": "task",
            "model_id": "smolvla_mi_human",
            "description": "put the trash in the dustbin",
            "metadata": "user_1_phase_1"
        }

        commands = [
            ("Task 1", task_cmd),
            # ("Start 1", {"command": "start", "metadata": "user_1_phase_1"}),
            # ("Stop 1", {"command": "stop"}),
        ]
        
        for name, cmd in commands:
            wait_for_enter(f"Press Enter to send '{name}'...")
            await send_command(ws, cmd)
        
        print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
