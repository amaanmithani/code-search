package server

import "fmt"

// Server holds listener state.
type Server struct {
	Port int
}

// Start opens the listening socket.
func (s *Server) Start() error {
	fmt.Println(s.Port)
	return nil
}

func NewServer(port int) *Server {
	return &Server{Port: port}
}
