package service

import (
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

var messageChannel = make(chan string)
var responseChannel = make(chan string)
var wsConn *websocket.Conn
var mu sync.Mutex

type ResponseMessage struct {
	Type        string `json:"type"`
	StreamStart bool   `json:"stream_start,omitempty"`
	Chunk       string `json:"chunk,omitempty"`
	StreamEnd   bool   `json:"stream_end,omitempty"`
	Error       string `json:"error,omitempty"`
}

// func Init() {
// 	fmt.Println("Init service")

// 	url := "ws://localhost:8765"
// 	var err error
// 	wsConn, _, err = websocket.DefaultDialer.Dial(url, nil)
// 	if err != nil {
// 		log.Fatal("dial:", err)
// 	}

// 	done := make(chan struct{})

// 	go func() {
// 		defer close(done)
// 		for {
// 			_, message, err := wsConn.ReadMessage()
// 			if err != nil {
// 				log.Println("read:", err)
// 				return
// 			}
// 			handleResponse(message)
// 		}
// 	}()

// 	go MessageProcessor(done)

// 	log.Println("Init service done")
// }

var (
	wsMutex sync.Mutex
)

const (
	maxReconnectAttempts = 5
	reconnectDelay       = 5 * time.Second
	pingPeriod           = 5 * time.Second  // Reduced from 10s
	pongWait             = 25 * time.Second // Reduced from 15s
	writeWait            = 10 * time.Second
)

func Init() {
	fmt.Println("Init service")

	url := "ws://localhost:8765"
	go connectWebSocket(url)

	log.Println("Init service done")
}

func connectWebSocket(url string) {
	for {
		wsConn = nil
		err := connect(url)
		if err != nil {
			log.Println("Connection failed:", err)
			time.Sleep(reconnectDelay)
			continue
		}

		log.Println("WebSocket connected")

		done := make(chan struct{})
		go readPump(done)
		go writePump(done)
		go MessageProcessor(done)

		<-done // Wait for done signal
		log.Println("WebSocket disconnected, attempting to reconnect...")
	}
}

func connect(url string) error {
	wsConn = nil
	wsDialer := websocket.Dialer{
		HandshakeTimeout: 5 * time.Second,
	}
	conn, _, err := wsDialer.Dial(url, nil)
	if err != nil {
		return err
	}
	wsConn = conn
	return nil
}

func readPump(done chan struct{}) {
	defer close(done)
	wsConn.SetReadLimit(512) // Limit size of incoming messages
	wsConn.SetReadDeadline(time.Now().Add(pongWait))
	wsConn.SetPongHandler(func(string) error {
		wsConn.SetReadDeadline(time.Now().Add(pongWait))
		return nil
	})
	for {
		_, message, err := wsConn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("error: %v", err)
			}
			return
		}
		handleResponse(message)
	}
}

func writePump(done chan struct{}) {
	ticker := time.NewTicker(pingPeriod)
	defer func() {
		ticker.Stop()
		wsConn.Close() // Ensure connection is closed
	}()

	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			if err := wsConn.WriteControl(websocket.PingMessage, []byte{}, time.Now().Add(writeWait)); err != nil {
				log.Println("ping error:", err)
				return
			}
		}
	}
}

func MessageProcessor(done chan struct{}) {
	log.Println("MessageProcessor started")
	for {
		select {
		case <-done:
			log.Println("MessageProcessor: WebSocket disconnected")
			return
		// case <-interrupt:
		// 	log.Println("interrupt")
		// 	mu.Lock()
		// 	err := wsConn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		// 	mu.Unlock()
		// 	if err != nil {
		// 		log.Println("write close:", err)
		// 	}
		// 	<-done
		// 	return
		case message := <-messageChannel:
			log.Println("Sending message (inside msg processor):", message)
			mu.Lock()
			wsConn.SetWriteDeadline(time.Now().Add(writeWait))
			wsConn.WriteMessage(websocket.TextMessage, []byte(message))
			mu.Unlock()

		}
	}
}

func Send(action string, message string, callback func(string, bool)) {
	log.Println("Send message:", message)

	if action == "generate" {
		message = fmt.Sprintf(`{"action": "GENERATE_RESPONSE", "prompt": "%s", "stream": true}`, message)
	} else {
		message = fmt.Sprintf(`{"action": "%s", "text": "%s"}`, action, message)
	}

	wsMutex.Lock()
	defer wsMutex.Unlock()

	if wsConn == nil {
		log.Println("WebSocket connection is nil")
		return
	}

	messageChannel <- message

	// The actual response handling is now done in the handleResponse function
	for {
		select {
		case response := <-responseChannel:
			callback(response, true)
			if response == "<EOR>" {
				return
			}
		}
	}
}

func handleResponse(message []byte) {
	var response ResponseMessage
	err := json.Unmarshal(message, &response)
	if err != nil {
		log.Println("Error unmarshalling response:", err)
		return
	}

	switch {
	case response.StreamStart:
		log.Println("Stream started")
	case response.Chunk != "":
		log.Printf("Received chunk: %s", response.Chunk)
		responseChannel <- string(message)
		// Here you would call your callback function with the chunk
		// callback(response.Chunk, false)
	case response.StreamEnd:
		log.Println("Stream ended")
		responseChannel <- string("<EOR>")
		// Here you would call your callback function to indicate the end of the stream
		// callback("", true)
	case response.Error != "":
		log.Printf("Error received: %s", response.Error)
	default:
		log.Printf("Unknown response type: %+v", response)
	}
}
