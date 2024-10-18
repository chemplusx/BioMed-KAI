package controller

import (
	"io"
	"log"
	"net/http"

	"chemplusx.com/midas/service"
	"github.com/gin-gonic/gin"
)

// MIDASController is the controller for the MIDAS service.

// MIDASHandler handles the MIDAS service.
func MIDASHandler(c *gin.Context) {
	// Parse the request
	var req service.MIDASRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Validate the request
	if err := req.Validate(); err != nil {
		log.Printf("Invalid request: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// Set headers for streaming
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	var respChan = make(chan string)
	go service.Send("generate", req.Text, func(response string, err bool) {
		log.Println("Sending response:", response)
		respChan <- response
	})

	c.Stream(func(w io.Writer) bool {
		// Generate and send data based on the request
		response := <-respChan
		c.SSEvent("message", response)
		c.Writer.Flush()
		// if err != nil {
		// 	log.Printf("Error generating response: %v", err)
		// 	c.SSEvent("error", fmt.Sprintf("Error: %v", err))
		// 	return false
		// }

		return true
	})

	// if f, ok := c.Writer.(http.Flusher); ok {
	// 	// Generate and send data based on the request

	// 	service.Send("generate", req.Text, func(response string) {
	// 		log.Println("Received response:", response)
	// 		c.SSEvent("message", response)
	// 		f.Flush()
	// 	})

	// 	// for i := 1; i <= req.Count; i++ {
	// 	// 	response := map[string]interface{}{
	// 	// 		"message": fmt.Sprintf("%s - %d", req.Text, i),
	// 	// 		"count":   i,
	// 	// 	}
	// 	// 	jsonResponse, _ := json.Marshal(response)
	// 	// 	c.SSEvent("message", string(jsonResponse))
	// 	// 	f.Flush()
	// 	// 	time.Sleep(1 * time.Second) // Simulate processing time
	// 	// }
	// } else {
	// 	c.AbortWithError(http.StatusInternalServerError, fmt.Errorf("Streaming unsupported"))
	// }
}
