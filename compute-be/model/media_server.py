import json
import websockets
from typing import Dict, List, Union
from .llama_if import LlamaCppModel
import asyncio

class MediaServer:
    def __init__(self):
        self.model = LlamaCppModel('llama-3.1')
        self.active_model = "llama-3.1"

        self.model, _ = self.model.load()

    async def handle_generate_response(self, data: Dict[str, Union[str, bool, List]], websocket: websockets.WebSocketServerProtocol):
        prompt = data.get("prompt", "")
        chat_history = data.get("chat_history", [])
        stream = data.get("stream", False)
        response_type = data.get("type", "ai")

        if response_type == "sourced":
            # Handle sourced response (implementation depends on your specific needs)
            response = "Sourced response: Not implemented in this example"
            await websocket.send(json.dumps({"type": "GENERATE_RESPONSE", "response": response}))
        else:
            # if stream:
            await websocket.send(json.dumps({"type": "GENERATE_RESPONSE", "stream_start": True}))
            
            async for chunk in self.model.generate(prompt, chat_history):
                await websocket.send(json.dumps({"type": "GENERATE_RESPONSE", "chunk": chunk}))
                print(f"Sent chunk: {chunk}")
                await asyncio.sleep(0)
            await websocket.send(json.dumps({"type": "GENERATE_RESPONSE", "stream_end": True}))
            # else:
            #     response = self.model.generate(prompt, chat_history)
            #     await websocket.send(json.dumps({"type": "GENERATE_RESPONSE", "response": response}))

    async def handle_active_model(self, websocket: websockets.WebSocketServerProtocol):
        await websocket.send(json.dumps({"type": "ACTIVE_MODEL", "model": self.active_model}))

    async def handle_select_model(self, data: Dict[str, str], websocket: websockets.WebSocketServerProtocol):
        new_model = data.get("name")
        if new_model in LlamaCppModel.get_models():
            self.active_model = new_model
            await websocket.send(json.dumps({"type": "SELECT_MODEL", "success": True, "model": new_model}))
        else:
            await websocket.send(json.dumps({"type": "SELECT_MODEL", "success": False, "error": "Model not found"}))

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        try:
            async for message in websocket:
                data = json.loads(message)
                action = data.get("action")

                if action == "GENERATE_RESPONSE":
                    await self.handle_generate_response(data, websocket)
                elif action == "ACTIVE_MODEL":
                    await self.handle_active_model(websocket)
                elif action == "SELECT_MODEL":
                    await self.handle_select_model(data, websocket)
                else:
                    await websocket.send(json.dumps({"type": "ERROR", "message": "Unknown action"}))
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")