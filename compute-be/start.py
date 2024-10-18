import asyncio
import json
import websockets
from typing import Dict, List, Union

from model.media_server import MediaServer

async def main():
    server = MediaServer()
    async with websockets.serve(server.handle_client, "0.0.0.0", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())