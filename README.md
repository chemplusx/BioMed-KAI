# BIOMed-KAI
Biomedicine Knowledge-Aided Intelligence


BIOMed-KAI is a context-aware precision medicine platform powered by Gen-AI and a dynamic medical knowledge graph, designed to support healthcare professionals in diagnostic decision-making and treatment planning.

## Features

- **Gen-AI Integration**: Advanced artificial intelligence for complex medical data analysis
- **Dynamic Knowledge Graph**: Continuously updated medical knowledge base
- **Precision Diagnostics**: Tailored diagnostic suggestions and treatment recommendations
- **Real-time Updates**: Regular integration of latest medical research and guidelines

## Tech Stack

- Frontend: VanillaJS + Tailwind CSS
- Knowledge Graph: Neo4j
- AI Engine: LlaMA 3.1
- Deployment: Docker containers

## Prerequisites

- Node.js >= 16.x
- Docker >= 20.x
- Neo4j >= 5.x
- Python >= 3.9
- golang >= 1.22.x

## Installation

```bash
# Clone the repository
git clone https://github.com/chemplusx/BioMed-KAI.git

# Install dependencies
cd BioMed-KAI
cd compute
pip install -r requirements.txt

```


## Development

In order to run BIOMed-KAI, you would need to run both the backend and the frontend servers.

```bash
# Running the backend Server with AI-Engine
cd compute
python start.py

# Running the frontend
cd webapp
go run main.go
OR
go build main.go && ./main
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Security

For security concerns, please email security@medal-ai.com

## License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md)
