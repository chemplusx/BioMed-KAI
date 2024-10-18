package main

import (
	"log"

	"chemplusx.com/midas/server"
	"chemplusx.com/midas/service"

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
	r.Run(":5050")
}
