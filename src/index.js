const express = require('express');
const cors = require('cors');
const { getConfig } = require('./config/config');
const FMPClient = require('./api/fmp');
const QuandlClient = require('./api/quandl');
const RiskCalculator = require('./calculators/RiskCalculator');
const { FinancialData, RiskMetrics } = require('./models/FinancialData');

const app = express();
const config = getConfig();

// Middleware
app.use(cors(config.security.cors));
app.use(express.json());

// Initialize API clients
const fmpClient = new FMPClient();
const quandlClient = new QuandlClient();
const riskCalculator = new RiskCalculator();

// Routes
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

app.get('/api/risk/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { period = '1Y' } = req.query;

    // Get historical prices
    const priceData = await fmpClient.getStockPrice(symbol);
    const financialData = await fmpClient.getFinancialStatements(symbol);

    // Calculate risk metrics
    // Note: This is a simplified example - you'd need historical price data
    const riskMetrics = new RiskMetrics({
      symbol: symbol,
      calculationDate: new Date(),
      timeHorizon: period
    });

    res.json({
      symbol: symbol,
      riskMetrics: riskMetrics.toJSON(),
      priceData: priceData,
      calculatedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error(`Error calculating risk for ${req.params.symbol}:`, error);
    res.status(500).json({
      error: 'Failed to calculate risk metrics',
      message: error.message
    });
  }
});

app.get('/api/economic-indicators', async (req, res) => {
  try {
    const { indicator = 'GDP', country = 'US' } = req.query;

    const [fmpData, quandlData] = await Promise.all([
      fmpClient.getEconomicIndicators(indicator),
      quandlClient.getMacroeconomicData('FRED', indicator)
    ]);

    res.json({
      indicator: indicator,
      country: country,
      fmpData: fmpData,
      quandlData: quandlData,
      retrievedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching economic indicators:', error);
    res.status(500).json({
      error: 'Failed to fetch economic indicators',
      message: error.message
    });
  }
});

app.get('/api/treasury-yields', async (req, res) => {
  try {
    const treasuryData = await fmpClient.getTreasuryYields();

    res.json({
      treasuryYields: treasuryData,
      retrievedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error fetching treasury yields:', error);
    res.status(500).json({
      error: 'Failed to fetch treasury yields',
      message: error.message
    });
  }
});

app.get('/api/esg/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;

    const esgData = await quandlClient.getESGScores(symbol);

    res.json({
      symbol: symbol,
      esgData: esgData,
      retrievedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error(`Error fetching ESG data for ${req.params.symbol}:`, error);
    res.status(500).json({
      error: 'Failed to fetch ESG data',
      message: error.message
    });
  }
});

app.get('/api/portfolio/risk', async (req, res) => {
  try {
    const { symbols, weights } = req.query;

    if (!symbols || !weights) {
      return res.status(400).json({
        error: 'Symbols and weights are required'
      });
    }

    const symbolArray = symbols.split(',');
    const weightArray = weights.split(',').map(w => parseFloat(w));

    if (symbolArray.length !== weightArray.length) {
      return res.status(400).json({
        error: 'Number of symbols must match number of weights'
      });
    }

    // Get data for each symbol and calculate portfolio risk
    const portfolioData = await Promise.all(
      symbolArray.map(async (symbol) => {
        const priceData = await fmpClient.getStockPrice(symbol);
        return { symbol, priceData };
      })
    );

    res.json({
      portfolio: portfolioData,
      weights: weightArray,
      calculatedAt: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error calculating portfolio risk:', error);
    res.status(500).json({
      error: 'Failed to calculate portfolio risk',
      message: error.message
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    path: req.path
  });
});

// Start server
const PORT = config.app.port;
app.listen(PORT, () => {
  console.log(`Risk Calculation Data Server running on port ${PORT}`);
  console.log(`Environment: ${config.app.env}`);
  console.log(`API endpoints available at http://localhost:${PORT}/api/`);
});