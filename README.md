# Risk Calculation Data Platform

A comprehensive financial risk calculation platform that integrates real-time data from Financial Modeling Prep (FMP) and Quandl APIs to provide advanced risk analytics, economic indicators, and portfolio risk assessment.

## 🚀 Features

- **Real-time Financial Data Integration**
  - Financial Modeling Prep API for earnings transcripts, treasury yields, risk premiums
  - Quandl API for macroeconomic data, ESG scores, central bank forecasts

- **Advanced Risk Calculations**
  - Value at Risk (VaR) and Expected Shortfall (ES)
  - Beta calculation and portfolio volatility
  - Sharpe ratio and maximum drawdown analysis
  - Portfolio risk assessment with correlation matrices

- **Comprehensive Data Models**
  - Structured financial data models
  - ESG scoring and sustainability metrics
  - Economic indicators and treasury data
  - Portfolio holdings and risk metrics

## 📁 Project Structure

```
risk-calculation-data/
├── src/
│   ├── api/              # API integration modules
│   │   ├── fmp.js        # Financial Modeling Prep client
│   │   └── quandl.js     # Quandl API client
│   ├── models/           # Data models and schemas
│   │   └── FinancialData.js
│   ├── calculators/      # Risk calculation algorithms
│   │   └── RiskCalculator.js
│   ├── config/           # Configuration management
│   │   └── config.js
│   ├── utils/            # Utility functions
│   └── index.js          # Main application entry point
├── config/               # Configuration files
├── tests/                # Test suites
├── docs/                 # Documentation
├── .env.example          # Environment variables template
├── package.json          # Dependencies and scripts
└── README.md            # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- API keys for FMP and Quandl

### Step-by-Step Setup

1. **Clone and Navigate to Project**
   ```bash
   cd risk-calculation-data
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Environment Configuration**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` file with your API keys:
   ```env
   FMP_API_KEY=your_fmp_api_key_here
   QUANDL_API_KEY=your_quandl_api_key_here
   PORT=3000
   NODE_ENV=development
   ```

4. **Start the Application**
   ```bash
   # Development mode with auto-reload
   npm run dev

   # Production mode
   npm start
   ```

## 📊 API Documentation

### Base URL
```
http://localhost:3000/api
```

### Health Check
```http
GET /health
```

### Risk Analysis Endpoints

#### Get Risk Metrics for Symbol
```http
GET /api/risk/{symbol}?period=1Y
```

**Parameters:**
- `symbol` (required): Stock symbol (e.g., AAPL, MSFT)
- `period` (optional): Time period for analysis (1Y, 6M, 3M)

**Response:**
```json
{
  "symbol": "AAPL",
  "riskMetrics": {
    "beta": 1.25,
    "volatility": 0.28,
    "sharpeRatio": 1.15,
    "valueAtRisk": -0.045,
    "expectedShortfall": -0.067,
    "maxDrawdown": 0.32
  },
  "calculatedAt": "2024-01-15T10:30:00Z"
}
```

#### Portfolio Risk Assessment
```http
GET /api/portfolio/risk?symbols=AAPL,MSFT,GOOGL&weights=0.4,0.3,0.3
```

### Economic Data Endpoints

#### Economic Indicators
```http
GET /api/economic-indicators?indicator=GDP&country=US
```

#### Treasury Yields
```http
GET /api/treasury-yields
```

#### ESG Scores
```http
GET /api/esg/{symbol}
```

## 🔧 Core Modules & Methods

### FMP Client (`src/api/fmp.js`)

**Key Methods:**
- `getEarningsTranscripts(symbol, year, quarter)` - Retrieve earnings call transcripts
- `getEconomicIndicators(indicator)` - Fetch economic indicators
- `getTreasuryYields()` - Get current treasury yield data
- `getRiskPremium(symbol)` - Calculate risk premium for symbol
- `getStockPrice(symbol)` - Real-time stock price data
- `getFinancialStatements(symbol, type, period)` - Company financial statements

### Quandl Client (`src/api/quandl.js`)

**Key Methods:**
- `getMacroeconomicData(dataset, code, startDate, endDate)` - Macroeconomic datasets
- `getESGScores(symbol)` - Environmental, Social, Governance scores
- `getCentralBankForecasts(country, indicator)` - Central bank economic forecasts
- `getInflationData(country)` - Inflation rate data
- `getCommodityPrices(commodity)` - Commodity price data
- `getCurrencyExchangeRates(from, to)` - Foreign exchange rates

### Risk Calculator (`src/calculators/RiskCalculator.js`)

**Core Calculation Methods:**
- `calculateVolatility(prices)` - Historical volatility calculation
- `calculateBeta(stockPrices, marketPrices)` - Beta coefficient
- `calculateSharpeRatio(prices, riskFreeRate)` - Risk-adjusted returns
- `calculateValueAtRisk(prices, confidenceLevel, timeHorizon)` - VaR calculation
- `calculateExpectedShortfall(prices, confidenceLevel)` - Expected Shortfall (CVaR)
- `calculateMaxDrawdown(prices)` - Maximum drawdown analysis
- `calculatePortfolioRisk(holdings, correlationMatrix)` - Portfolio-level risk

## 📈 Risk Calculation Procedures

### 1. Data Collection Process
1. **Price Data Retrieval**: Fetch historical price data from FMP API
2. **Market Data Integration**: Collect market index data for beta calculations
3. **Economic Context**: Retrieve relevant economic indicators from Quandl
4. **Data Validation**: Ensure data quality and completeness

### 2. Risk Metrics Calculation Workflow
1. **Return Calculation**: Convert price series to return series
2. **Volatility Analysis**: Calculate annualized volatility using standard deviation
3. **Beta Calculation**: Measure systematic risk relative to market
4. **VaR Computation**: Historical simulation method for Value at Risk
5. **Expected Shortfall**: Conditional VaR for tail risk assessment

### 3. Portfolio Risk Assessment Steps
1. **Holdings Analysis**: Parse portfolio weights and allocations
2. **Correlation Matrix**: Calculate pairwise correlations between assets
3. **Portfolio Volatility**: Weighted portfolio risk calculation
4. **Diversification Benefits**: Measure risk reduction from diversification

## 🧪 Testing

```bash
# Run all tests
npm test

# Run tests with coverage
npm run test:coverage

# Run linting
npm run lint
```

## 📝 Usage Examples

### Basic Risk Analysis
```javascript
const FMPClient = require('./src/api/fmp');
const RiskCalculator = require('./src/calculators/RiskCalculator');

const fmp = new FMPClient();
const riskCalc = new RiskCalculator();

// Get stock data and calculate risk metrics
const stockData = await fmp.getStockPrice('AAPL');
const riskMetrics = riskCalc.calculateRiskMetrics('AAPL', stockData);
```

### Portfolio Risk Assessment
```javascript
const portfolio = [
  { symbol: 'AAPL', weight: 0.4, volatility: 0.25 },
  { symbol: 'MSFT', weight: 0.3, volatility: 0.22 },
  { symbol: 'GOOGL', weight: 0.3, volatility: 0.28 }
];

const correlationMatrix = [
  [1.0, 0.7, 0.6],
  [0.7, 1.0, 0.8],
  [0.6, 0.8, 1.0]
];

const portfolioRisk = riskCalc.calculatePortfolioRisk(portfolio, correlationMatrix);
```

## 🔐 Security & Rate Limits

- **API Rate Limits**:
  - FMP: 250 requests/minute, 5,000/day
  - Quandl: 300 requests/minute, 50,000/day
- **Environment Variables**: Secure API key management
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Input Validation**: Request parameter validation and sanitization

## 🚀 Deployment

### Environment Setup
1. Set production environment variables
2. Configure database connections (if applicable)
3. Set up monitoring and logging
4. Configure reverse proxy (nginx/Apache)

### Production Commands
```bash
# Install production dependencies
npm ci --production

# Start production server
NODE_ENV=production npm start
```

## 📊 Progress & Development Status

### ✅ Completed Features
- [x] Project structure and configuration
- [x] FMP API integration with comprehensive endpoints
- [x] Quandl API integration for economic data
- [x] Complete data models for financial data types
- [x] Advanced risk calculation algorithms
- [x] RESTful API endpoints for risk analysis
- [x] Configuration management and environment setup
- [x] Comprehensive documentation

### 🔄 In Progress
- [ ] Unit test suite implementation
- [ ] Database integration for data persistence
- [ ] Caching layer for improved performance
- [ ] WebSocket support for real-time updates

### 📋 Future Enhancements
- [ ] Machine learning risk models
- [ ] Advanced portfolio optimization
- [ ] Risk reporting and visualization
- [ ] Multi-currency support
- [ ] Historical backtesting capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request
