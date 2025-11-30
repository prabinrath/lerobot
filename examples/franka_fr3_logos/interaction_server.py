"""WebSocket server for interactive robot control via ROS2 pub/sub.

Commands:
- {"command": "task", ...} -> publishes to '/rollout_command'
- {"command": "start"} -> publishes "start" to '/record_command'
- {"command": "stop"} -> publishes "stop" to '/record_command'
"""

import asyncio
import json
import logging
import threading

import websockets
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class InteractivePublisher(Node):
    def __init__(self):
        super().__init__('interactive_publisher')
        self._rollout_pub = self.create_publisher(String, '/rollout_command', 10)
        self._record_pub = self.create_publisher(String, '/record_command', 10)
    
    def publish_rollout(self, data: dict):
        msg = String()
        msg.data = json.dumps(data)
        self._rollout_pub.publish(msg)
        logger.info(f"Rollout: {data}")
    
    def publish_record(self, command: str):
        msg = String()
        msg.data = command
        self._record_pub.publish(msg)
        logger.info(f"Record: {command}")


async def main():
    rclpy.init()
    node = InteractivePublisher()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    
    logger.info("Starting WebSocket server on ws://0.0.0.0:8765")
    
    async def handler(ws):
        async for msg in ws:
            data = json.loads(msg)
            cmd = data.get("command")
            
            if cmd == "task":
                if not data.get("model_id") or not data.get("description") or not data.get("metadata"):
                    resp = {"status": "error", "message": "Missing model_id or description or metadata"}
                else:
                    node.publish_rollout(data)
                    resp = {"status": "ok", "message": "Rollout command published"}
            elif cmd in ("start", "stop"):
                node.publish_record(cmd)
                resp = {"status": "ok", "message": f"Record {cmd} published"}
            else:
                resp = {"status": "error", "message": "Unknown command"}
            
            await ws.send(json.dumps(resp))
    
    async with websockets.serve(handler, "0.0.0.0", 8765, ping_timeout=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())