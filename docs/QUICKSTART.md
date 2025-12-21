# Quick Start Guide
Get Supreme System V5 running in 15 minutes

## ⚡ Prerequisites

- **Python**: 3.10+ (3.11+ recommended)
- **Docker**: 24.0+ with Docker Compose (for production)
- **Git**: Latest version
- **System**: 4GB RAM, 2 CPU cores minimum

## 📦 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/thanhmuefatty07/supreme-system-v5.git
cd supreme-system-v5
```

### Step 2: Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Step 3: Configure Settings

Edit `.env` file:

```bash
# Essential settings
TRADING_MODE=paper          # Start with paper trading
SYMBOL_FOCUS=AAPL           # Your trading symbol
RISK_LEVEL=minimal          # Conservative risk

# Optional: Add API keys for live data
# BINANCE_API_KEY=your_key_here
# BINANCE_SECRET_KEY=your_secret_here
```

### Step 4: Verify Installation

```bash
# Run tests to verify setup
pytest tests/ -v

# Should see: X tests passed
```

## 🎮 Basic Usage

### Run Paper Trading Simulation

```bash
python -m src.cli paper-trade \
    --symbols AAPL MSFT GOOGL \
    --capital 100000 \
    --strategy momentum
```

### Run Backtesting

```bash
python -m src.cli backtest \
    --strategy momentum \
    --symbols AAPL \
    --days 365 \
    --output results/backtest_report.json
```

### Start Dashboard

```bash
# Interactive web dashboard
streamlit run src/monitoring/dashboard.py

# Open browser: http://localhost:8501
```

## 🐳 Docker Deployment (Recommended for Production)

### Quick Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Services

- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 📊 Monitoring & Metrics

### View System Metrics

```bash
# Check system health
curl http://localhost:8000/health

# View metrics
curl http://localhost:9090/metrics
```

### Grafana Dashboards

1. Login to Grafana (admin/admin)
2. Import dashboard from `monitoring/grafana/dashboards/`
3. View real-time trading metrics

## 🔍 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Port Already in Use**:
```bash
# Change ports in docker-compose.yml or .env
```

**API Connection Errors**:
```bash
# Verify API keys in .env
# Check network connectivity
# Use EXCHANGE_SANDBOX=true for testing
```

## 📚 Next Steps

1. **Customize Strategies**: See `docs/STRATEGIES.md`
2. **Configure Risk**: See `docs/RISK_MANAGEMENT.md`
3. **Production Deployment**: See `docs/DEPLOYMENT.md`
4. **API Integration**: See `docs/API.md`

## 🆘 Support

- **Documentation**: `/docs` folder
- **Email**: thanhmuefatty07@gmail.com
- **Issues**: GitHub Issues (for licensed users)

---
**Tip**: Start with paper trading and gradually increase to live trading only after thorough testing.

**Last Updated**: November 16, 2025
