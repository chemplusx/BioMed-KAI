import json
import websockets
from typing import Dict, List, Union, Set
from .llama_if import LlamaCppModel
import asyncio

class MediaServer:
    def __init__(self):
        print("Calling - > ", LlamaCppModel.get_models())
        self.active_model = "llama-3.1"
        self.model, _ = LlamaCppModel(self.active_model).load()
        self.connected_clients: Set[websockets.WebSocketServerProtocol] = set()

    async def handle_connect(self, websocket: websockets.WebSocketServerProtocol):
        """Handle new client connections"""
        self.connected_clients.add(websocket)
        # Send initial state to the client
        await websocket.send(json.dumps({
            "type": "CONNECTION_ESTABLISHED",
            "data": {
                "activeModel": self.active_model,
                "availableModels": LlamaCppModel.get_models(),
                "clientId": id(websocket)  # Using object id as a simple unique identifier
            }
        }))

    async def handle_disconnect(self, websocket: websockets.WebSocketServerProtocol):
        """Handle client disconnections"""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
            print(f"Client {id(websocket)} disconnected. Active connections: {len(self.connected_clients)}")

    async def broadcast_message(self, message: Dict):
        """Broadcast a message to all connected clients"""
        if self.connected_clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self.connected_clients]
            )

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        try:
            await self.handle_connect(websocket)
            print(f"New client connected. Active connections: {len(self.connected_clients)}")

            async for message in websocket:
                data = json.loads(message)
                action = data.get("action")

                if action == "GENERATE_RESPONSE":
                    await self.handle_generate_response(data, websocket)
                elif action == "ACTIVE_MODEL":
                    await self.handle_active_model(websocket)
                elif action == "SELECT_MODEL":
                    await self.handle_select_model(data, websocket)
                    # Broadcast model change to all clients
                    await self.broadcast_message({
                        "type": "MODEL_CHANGED",
                        "model": self.active_model
                    })
                else:
                    await websocket.send(json.dumps({
                        "type": "ERROR",
                        "message": "Unknown action"
                    }))

        except websockets.exceptions.ConnectionClosed:
            print(f"Client {id(websocket)} connection closed unexpectedly")
        except Exception as e:
            print(f"Error handling client {id(websocket)}: {str(e)}")
        finally:
            await self.handle_disconnect(websocket)

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
            
            # async for chunk in self.model.generate(prompt, chat_history):
            async for chunk in self.model.generate2(prompt, chat_history):
                # print(f"Chunk: {chunk}")
                await websocket.send(json.dumps({"type": "GENERATE_RESPONSE", "chunk": chunk}))
                # print(f"Sent chunk: {chunk}")
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
