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
        # asyncio.create_task(self.runTest(test="custom"))

    async def runTest(self, test="all"):
        print("Running Test")
        # Read the testing file
        if test == "all" or test == "medqa":
            with open("H:\\workspace\\nameless-ai\\compute-be\\metrics\\data\\medqa.jsonl", "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    data = json.loads(line)
                    response = await self.model.generate1(
                        "Answer the question by correctly choosing from the given options. Question: " + data["question"] + "-> Options: " + str(data["options"]), 
                        []
                    )
                    print("Response: ", response)

                    # Save the response to the file
                    with open("H:\\workspace\\nameless-ai\\compute-be\\metrics\\data\\medqa_responses1.jsonl", "a") as file:
                        file.write(json.dumps({
                            "question": data["question"],
                            "answer": data["answer"],
                            "response": response
                        }) + "\n")
        if test == "all" or test == "medmcqa":
            with open("H:\\workspace\\nameless-ai\\compute-be\\metrics\\data\\medmcqa.json", "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    """
                    {
                        "question": "Which of the following are not a branch of external carotid Aery in Kiesselbach's plexus.",
                        "exp": "*Kiesselbach's plexus: Antero superior pa is supplied by ANTERIOR & POSTERIOR ETHMOIDAL AERIES which are branches of ophthalmic aery, branch of INTERNAL CAROTID AERY. Antero inferior pa is supplied by SUPERIOR LABIAL AERY - branch of facial aery, which is branch of EXTERNAL CAROTID AERY. Postero superior pa is supplied by SPHENO-PALATINE AERY - branch of MAXILLARY aery, which is branch of ECA. POSTERO INFERIOR pa is supplied by branches of GREATER PALATINE AERY - branch of ECA Antero inferior pa\/vestibule of septum contain anastomosis b\/w septal ramus of superior labial branch of facial aery & branches of sphenopalatine, greater palatine & anterior ethmoidal aeries. These form a large capillary network called KIESSELBACH'S PLEXUS If dryness persists, bleeding will occur Therefore, in given options, Anterior ethmoidal aery is a branch of ICA not ECA",
                        "cop": 2,
                        "opa": "Sphenopalatine aery",
                        "opb": "Anterior ethmoidal aery",
                        "opc": "Greater palatine aery",
                        "opd": "Septal branch of superior labial aery",
                        "subject_name": "Anatomy",
                        "topic_name": "AIIMS 2017",
                        "id": "ce49098b-cc48-4168-859e-936e3e0c7459",
                        "choice_type": "single"
                    }
                    """
                    data = json.loads(line)
                    response = await self.model.generate1(
                        "Answer the question by correctly choosing from the given options. Question: " + data["question"] + ". Options: a) " + data["opa"] + ", b) " + data["opb"] + ", c) " + data["opc"] + ", d) " + data["opd"],
                        []
                    )
                    print("Response: ", response)

                    # Save the response to the file
                    if data["cop"] == 1:
                        answer = data["opa"]
                    elif data["cop"] == 2:
                        answer = data["opb"]
                    elif data["cop"] == 3:
                        answer = data["opc"]
                    else:
                        answer = data["opd"]
                    with open("H:\\workspace\\nameless-ai\\compute-be\\metrics\\data\\medmcqa_responses.jsonl", "a") as file:
                        file.write(json.dumps({
                            "question": data["question"],
                            "answer": answer,
                            "response": response
                        }) + "\n")
        if test == "all" or test == "medalpaca":
            # read medalpaca_flashcards.csv
            import pandas as pd
            df = pd.read_csv("H:\\workspace\\nameless-ai\\compute-be\\metrics\\data\\medalpaca_flashcards.csv")
            """
            sample data:
                "input","output","instruction"
                "What is the relationship between very low Mg2+ levels, PTH levels, and Ca2+ levels?","Very low Mg2+ levels correspond to low PTH levels which in turn results in low Ca2+ levels.","Answer this question truthfully"

            """
            for index, row in df.iterrows():
                response = await self.model.generate1(
                    "Answer the question in brief. Question: " + row["input"],
                    []
                )
                print("Response: ", response)

                # Save the response to the file
                with open("H:\\workspace\\nameless-ai\\compute-be\\metrics\\data\\medalpaca_responses.jsonl", "a") as file:
                    file.write(json.dumps({
                        "question": row["input"],
                        "answer": row["output"],
                        "response": response
                    }) + "\n")
        if test == "all" or test == "custom":
            # read custom_biomed.csv
            import pandas as pd
            df = pd.read_csv("H:\\workspace\\nameless-ai\\compute-be\\metrics\\data\\custom_biomed.csv")
            """
            sample data:
                Query,Subject,Expected Answer,Verified Source of expected answer,BioMedKai ,Result
            """
            for index, row in df.iterrows():
                response = await self.model.generate1(
                    "Answer the question in brief. Question: " + row["Query"],
                    []
                )
                print("Response: ", response)

                # Save the response to the file
                with open("H:\\workspace\\nameless-ai\\compute-be\\metrics\\data\\custom_biomed_responses.jsonl", "a") as file:
                    file.write(json.dumps({
                        "question": row["Query"],
                        "answer": row["Expected Answer"],
                        "response": response
                    }) + "\n")

                    
        # print(self.model.generate1("Hello", ["Hi", "Hello", "How are you?"]))


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
