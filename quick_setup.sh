#!/bin/bash

# RAG System Quick Setup Script
# This script automates the setup process

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
    echo ""
    echo "======================================"
    echo "$1"
    echo "======================================"
}

# Check if running from correct directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_header "RAG System Quick Setup"

# Step 1: Check prerequisites
print_header "Step 1: Checking Prerequisites"

# Check Docker
if command -v docker &> /dev/null; then
    print_success "Docker installed"
else
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_success "Docker Compose installed"
else
    print_error "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    print_success "Python 3 installed"
else
    print_error "Python 3 not found. Please install Python 3 first."
    exit 1
fi

# Step 2: Setup environment
print_header "Step 2: Setting Up Environment"

if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    cp .env.example .env
    print_success ".env file created"
    print_warning "Please review and update .env file with your settings"
else
    print_success ".env file already exists"
fi

# Step 3: Install Python dependencies
print_header "Step 3: Installing Python Dependencies"

cd backend
if [ -f "requirements.txt" ]; then
    print_info "Installing Python packages..."
    pip3 install -r requirements.txt > /dev/null 2>&1
    print_success "Python dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Step 4: Start Docker services
print_header "Step 4: Starting Docker Services"

cd ..
print_info "Starting Docker containers..."
docker-compose up -d

# Wait for services to be ready
print_info "Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    print_success "Docker services started"
else
    print_error "Failed to start Docker services"
    docker-compose logs
    exit 1
fi

# Step 5: Wait for PostgreSQL
print_header "Step 5: Waiting for PostgreSQL"

print_info "Waiting for PostgreSQL to be ready..."
MAX_TRIES=30
TRIES=0

while [ $TRIES -lt $MAX_TRIES ]; do
    if docker-compose exec -T postgres pg_isready -U rag_user -d rag_db > /dev/null 2>&1; then
        print_success "PostgreSQL is ready"
        break
    fi
    TRIES=$((TRIES+1))
    if [ $TRIES -eq $MAX_TRIES ]; then
        print_error "PostgreSQL failed to start"
        exit 1
    fi
    sleep 2
done

# Step 6: Initialize database
print_header "Step 6: Initializing Database"

cd backend
print_info "Creating database tables and indexes..."

if python3 init_db.py; then
    print_success "Database initialized successfully"
else
    print_error "Database initialization failed"
    exit 1
fi

# Step 7: Pull Ollama model
print_header "Step 7: Pulling Ollama Model"

print_info "Pulling Mistral 7B model (this may take a few minutes)..."
if docker exec rag_ollama ollama pull mistral:7b-instruct-q4_0; then
    print_success "Ollama model downloaded"
else
    print_warning "Failed to pull Ollama model. You can do this manually later:"
    print_warning "  docker exec rag_ollama ollama pull mistral:7b-instruct-q4_0"
fi

# Step 8: Verify setup
print_header "Step 8: Verifying Setup"

print_info "Running system verification..."
if python3 verify_setup.py; then
    print_success "All checks passed!"
else
    print_warning "Some checks failed. Review the output above."
fi

# Step 9: Summary
print_header "Setup Complete!"

echo ""
echo "Your RAG system is now ready to use!"
echo ""
echo "Access points:"
echo "  - Frontend:  http://localhost:3000"
echo "  - API:       http://localhost:8000"
echo "  - API Docs:  http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  - View logs:          docker-compose logs -f"
echo "  - Stop services:      docker-compose down"
echo "  - Restart services:   docker-compose restart"
echo "  - Verify system:      cd backend && python3 verify_setup.py"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:8000/docs to explore the API"
echo "  2. Upload a document using the /api/resources/upload endpoint"
echo "  3. Create a conversation and start chatting!"
echo ""

# Optional: Start log streaming
read -p "Would you like to view the logs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose logs -f
fi