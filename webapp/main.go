package main

import (
	"log"

	"chemplusx.com/medal-ai/server"
	"chemplusx.com/medal-ai/service"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	r.GET("/ping", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "pong",
		})
	})
	log.Println("Starting server")
	service.Init()
	log.Println("Server started")
	server.InitRouter(r)
	r.Run(":5000")
}
