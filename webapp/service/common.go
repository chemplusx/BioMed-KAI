package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

type Config struct {
	URL                  string
	MaxReconnectAttempts int
	ReconnectDelay       time.Duration
	PingPeriod           time.Duration
	PongWait             time.Duration
	WriteWait            time.Duration
	MessageBufferSize    int
}

type Service struct {
	config Config
	conn   *websocket.Conn
	mu     sync.RWMutex

	ctx    context.Context
	cancel context.CancelFunc

	messageChannel  chan Message
	responseChannel chan string
	responseDone    chan struct{}

	isConnected bool
}

type ConversationEntry struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

// Updated Message struct to match the new backend
type Message struct {
	Action    string                 `json:"action"`
	Prompt    string                 `json:"prompt,omitempty"`
	Text      string                 `json:"text,omitempty"`
	History   []ConversationEntry    `json:"history,omitempty"`
	Stream    bool                   `json:"stream,omitempty"`
	UserID    string                 `json:"user_id,omitempty"`
	SessionID string                 `json:"session_id,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Updated ResponseMessage struct
type ResponseMessage struct {
	Type      string                 `json:"type"`
	Status    string                 `json:"status,omitempty"`
	Content   string                 `json:"content,omitempty"`
	Delta     string                 `json:"delta,omitempty"`
	Done      bool                   `json:"done,omitempty"`
	Error     string                 `json:"error,omitempty"`
	MessageID string                 `json:"message_id,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Updated DefaultConfig to match new backend
func DefaultConfig() Config {
	return Config{
		URL:                  "ws://localhost:8080/ws/chat", // Updated URL
		MaxReconnectAttempts: 100,
		ReconnectDelay:       5 * time.Second,
		PingPeriod:           30 * time.Second, // Increased ping period
		PongWait:             60 * time.Second, // Increased pong wait
		WriteWait:            10 * time.Second,
		MessageBufferSize:    100,
	}
}

func NewService(config Config) *Service {
	ctx, cancel := context.WithCancel(context.Background())
	return &Service{
		config:          config,
		ctx:             ctx,
		cancel:          cancel,
		messageChannel:  make(chan Message, config.MessageBufferSize),
		responseChannel: make(chan string, config.MessageBufferSize),
		responseDone:    make(chan struct{}),
	}
}

func (s *Service) Start() error {
	log.Println("Starting WebSocket service")
	go s.connectionManager()
	return nil
}

func (s *Service) Stop() {
	log.Println("Stopping WebSocket service")
	s.cancel()
	s.closeConnection()
}

func (s *Service) connectionManager() {
	attempts := 0
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
			if err := s.connect(); err != nil {
				log.Printf("Connection failed: %v", err)
				s.setConnected(false)
				attempts++

				if s.config.MaxReconnectAttempts > 0 && attempts >= s.config.MaxReconnectAttempts {
					log.Printf("Max reconnection attempts (%d) reached", s.config.MaxReconnectAttempts)
					return
				}

				select {
				case <-s.ctx.Done():
					return
				case <-time.After(s.config.ReconnectDelay):
					continue
				}
			}

			attempts = 0

			errCh := make(chan error, 3)
			go s.readPump(errCh)
			go s.writePump(errCh)
			go s.messageProcessor(errCh)

			select {
			case err := <-errCh:
				log.Printf("Handler error: %v", err)
				s.closeConnection()
				continue
			case <-s.ctx.Done():
				return
			}
		}
	}
}

func (s *Service) connect() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}

	dialer := websocket.Dialer{
		HandshakeTimeout: 5 * time.Second,
	}

	conn, _, err := dialer.Dial(s.config.URL, nil)
	if err != nil {
		return fmt.Errorf("dial error: %w", err)
	}

	s.conn = conn
	s.isConnected = true
	log.Println("WebSocket connected successfully")
	return nil
}

func (s *Service) closeConnection() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.conn != nil {
		s.conn.Close()
		s.conn = nil
	}
	s.isConnected = false

	select {
	case s.responseDone <- struct{}{}:
	default:
	}
}

func (s *Service) readPump(errCh chan<- error) {
	defer func() {
		if r := recover(); r != nil {
			errCh <- fmt.Errorf("readPump panic recovered: %v", r)
		}
	}()

	s.conn.SetReadLimit(32768)
	s.conn.SetReadDeadline(time.Now().Add(s.config.PongWait))
	s.conn.SetPongHandler(func(string) error {
		s.conn.SetReadDeadline(time.Now().Add(s.config.PongWait))
		return nil
	})

	for {
		_, message, err := s.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				errCh <- fmt.Errorf("unexpected close error: %w", err)
			}
			return
		}
		if err := s.handleResponse(message); err != nil {
			log.Printf("Error handling response: %v", err)
		}
	}
}

func (s *Service) writePump(errCh chan<- error) {
	ticker := time.NewTicker(s.config.PingPeriod)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			if !s.IsConnected() {
				errCh <- fmt.Errorf("websocket is not connected")
				continue
			}
			s.mu.Lock()
			err := s.conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(s.config.WriteWait))
			s.mu.Unlock()
			if err != nil {
				errCh <- fmt.Errorf("ping error: %w", err)
				return
			}
		}
	}
}

func (s *Service) messageProcessor(errCh chan<- error) {
	defer func() {
		if r := recover(); r != nil {
			errCh <- fmt.Errorf("messageProcessor panic recovered: %v", r)
		}
	}()

	for {
		select {
		case <-s.ctx.Done():
			return
		case msg := <-s.messageChannel:
			log.Println("Processing message", msg)
			if err := s.writeMessage(msg); err != nil {
				errCh <- fmt.Errorf("write message error: %w", err)
				return
			}
		}
	}
}

func (s *Service) writeMessage(msg Message) error {
	writeDone := make(chan struct{})
	go func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		defer close(writeDone)

		if s.conn == nil {
			return
		}

		data, err := json.Marshal(msg)
		if err != nil {
			log.Printf("Marshal error: %v", err)
			return
		}

		log.Println("Writing message to WebSocket:", string(data))
		s.conn.SetWriteDeadline(time.Now().Add(s.config.WriteWait))
		if err := s.conn.WriteMessage(websocket.TextMessage, data); err != nil {
			log.Printf("Write error: %v", err)
			return
		}
	}()

	select {
	case <-writeDone:
		return nil
	case <-time.After(5 * time.Second):
		return fmt.Errorf("write message timeout")
	}
}

// Updated Send method
func (s *Service) Send(action string, data MIDASRequest, callback func(string, bool)) error {
	log.Println("Sending message to WebSocket")
	if !s.IsConnected() {
		return fmt.Errorf("websocket is not connected")
	}

	msg := Message{
		Action:    "chat", // Use 'chat' action for the new backend
		Prompt:    data.Text,
		Stream:    true,
		UserID:    data.Id,
		SessionID: fmt.Sprintf("session_%s", data.Id),
	}

	// Convert conversations to history
	if data.Conversations != nil && len(data.Conversations) > 0 {
		for _, c := range data.Conversations {
			parts := strings.Split(c, "-$$-")
			if len(parts) == 2 {
				msg.History = append(msg.History, ConversationEntry{
					Role: parts[0],
					Text: parts[1],
				})
			}
		}
	}

	select {
	case s.messageChannel <- msg:
	case <-time.After(5 * time.Second):
		return fmt.Errorf("timeout sending message")
	}

	go func() {
		for {
			select {
			case <-s.ctx.Done():
				callback("", true)
				return
			case <-s.responseDone:
				callback("Connection closed", true)
				return
			case response := <-s.responseChannel:
				if response == "<EOR>" {
					callback("", true)
					return
				}
				callback(response, false)
			case <-time.After(60 * time.Second): // Increased timeout
				callback("Response timeout", true)
				return
			}
		}
	}()

	return nil
}

type AnalysisRequest struct {
	Input     string   `json:"input"`
	Specialty string   `json:"specialty"`
	UserID    string   `json:"userId"`
	SessionID string   `json:"sessionId"`
	Files     []string `json:"files,omitempty"`
}

// Updated SendForAnalysis method
func (s *Service) SendForAnalysis(action string, data AnalysisRequest, callback func(string, bool)) error {
	log.Println("Sending analysis message to WebSocket")
	if !s.IsConnected() {
		return fmt.Errorf("websocket is not connected")
	}

	msg := Message{
		Action:    "chat", // Use 'chat' action for the new backend
		Prompt:    data.Input,
		Stream:    true,
		UserID:    data.UserID,
		SessionID: data.SessionID,
		Metadata: map[string]interface{}{
			"specialty": data.Specialty,
			"files":     data.Files,
			"type":      "analysis",
		},
	}

	select {
	case s.messageChannel <- msg:
	case <-time.After(5 * time.Second):
		return fmt.Errorf("timeout sending message")
	}

	go func() {
		for {
			select {
			case <-s.ctx.Done():
				callback("", true)
				return
			case <-s.responseDone:
				callback("Connection closed", true)
				return
			case response := <-s.responseChannel:
				if response == "<EOR>" {
					callback("", true)
					return
				}
				callback(response, false)
			case <-time.After(60 * time.Second): // Increased timeout
				callback("Response timeout", true)
				return
			}
		}
	}()

	return nil
}

// Updated handleResponse to work with new message format
func (s *Service) handleResponse(message []byte) error {
	var response ResponseMessage
	if err := json.Unmarshal(message, &response); err != nil {
		log.Printf("Unmarshal error: %v", err)
		return fmt.Errorf("unmarshal error: %w", err)
	}

	sendResponse := func(msg string) error {
		select {
		case s.responseChannel <- msg:
			return nil
		case <-time.After(5 * time.Second):
			return fmt.Errorf("timeout sending response")
		}
	}

	log.Println("Received response:", response)

	switch response.Type {
	case "stream_start":
		log.Println("Stream started")
	case "stream_delta":
		if response.Delta != "" {
			return sendResponse(response.Delta)
		}
	case "stream_end":
		return sendResponse("<EOR>")
	case "error":
		return fmt.Errorf("server error: %s", response.Error)
	default:
		// Handle content field for non-streaming responses
		if response.Content != "" {
			return sendResponse(response.Content)
		}
		// Handle done field
		if response.Done {
			return sendResponse("<EOR>")
		}
	}

	return nil
}

func (s *Service) IsConnected() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.isConnected
}

func (s *Service) setConnected(status bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.isConnected = status
}
