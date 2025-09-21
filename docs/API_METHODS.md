# API Methods and Procedures Documentation

## Table of Contents
1. [Financial Modeling Prep (FMP) API Methods](#fmp-api-methods)
2. [Quandl API Methods](#quandl-api-methods)
3. [Risk Calculation Methods](#risk-calculation-methods)
4. [Data Flow Procedures](#data-flow-procedures)
5. [Error Handling Procedures](#error-handling-procedures)

---

## FMP API Methods

### Core Market Data Methods

#### `getStockPrice(symbol)`
**Purpose**: Retrieve real-time stock price data
**Parameters**:
- `symbol` (string): Stock ticker symbol (e.g., 'AAPL')
**Returns**: Object containing current price, volume, market cap
**Rate Limit**: Contributes to 250 requests/minute limit
**Example**:
```javascript
const price = await fmpClient.getStockPrice('AAPL');
// Returns: { symbol: 'AAPL', price: 150.25, volume: 45000000, ... }
```

#### `getCompanyProfile(symbol)`
**Purpose**: Get comprehensive company information
**Parameters**:
- `symbol` (string): Stock ticker symbol
**Returns**: Company profile including sector, industry, description, executives
**Use Case**: Fundamental analysis and company categorization
**Example**:
```javascript
const profile = await fmpClient.getCompanyProfile('MSFT');
// Returns: { symbol: 'MSFT', companyName: 'Microsoft Corporation', sector: 'Technology', ... }
```

### Financial Statements Methods

#### `getFinancialStatements(symbol, type, period)`
**Purpose**: Retrieve company financial statements
**Parameters**:
- `symbol` (string): Stock ticker symbol
- `type` (string): 'income-statement', 'balance-sheet', 'cash-flow-statement'
- `period` (string): 'annual' or 'quarter'
**Returns**: Array of financial statement data
**Procedure**:
1. Validate symbol format
2. Make API request with proper parameters
3. Parse and structure financial data
4. Return standardized financial metrics

#### `getEarningsTranscripts(symbol, year, quarter)`
**Purpose**: Access earnings call transcripts for sentiment analysis
**Parameters**:
- `symbol` (string): Company ticker
- `year` (number, optional): Specific year
- `quarter` (number, optional): Quarter (1-4)
**Returns**: Transcript text and metadata
**Use Case**: Qualitative risk assessment through management commentary

### Economic Data Methods

#### `getEconomicIndicators(indicator)`
**Purpose**: Fetch macroeconomic indicators
**Parameters**:
- `indicator` (string): Economic indicator name ('GDP', 'INFLATION', etc.)
**Returns**: Historical economic data series
**Integration Point**: Links with Quandl data for cross-validation

#### `getTreasuryYields()`
**Purpose**: Current US Treasury yield curve data
**Parameters**: None
**Returns**: Yield data across different maturities
**Critical For**: Risk-free rate calculations in Sharpe ratio and CAPM

#### `getRiskPremium(symbol)`
**Purpose**: Calculate market risk premium for specific security
**Parameters**:
- `symbol` (string): Stock ticker
**Returns**: Risk premium calculation
**Method**: Market return minus risk-free rate

---

## Quandl API Methods

### Macroeconomic Data Methods

#### `getMacroeconomicData(dataset, code, startDate, endDate)`
**Purpose**: Access government and institutional economic datasets
**Parameters**:
- `dataset` (string): Data provider code (e.g., 'FRED', 'OECD')
- `code` (string): Specific indicator code
- `startDate` (string, optional): Start date for data range
- `endDate` (string, optional): End date for data range
**Returns**: Time series economic data
**Procedure**:
1. Construct dataset path: `${dataset}/${code}`
2. Add date filters if provided
3. Execute API request with authentication
4. Parse and validate returned data
5. Transform into standardized format

#### `getCentralBankForecasts(country, indicator)`
**Purpose**: Access central bank economic forecasts
**Parameters**:
- `country` (string): Country code ('US', 'EU', 'UK', etc.)
- `indicator` (string): Economic indicator ('GDP', 'INFLATION')
**Returns**: Forward-looking economic projections
**Use Case**: Incorporate expected economic conditions into risk models

### ESG and Sustainability Methods

#### `getESGScores(symbol)`
**Purpose**: Environmental, Social, Governance scoring data
**Parameters**:
- `symbol` (string): Company ticker
**Returns**: ESG scores and sustainability metrics
**Integration**: Links with risk calculations for ESG-adjusted risk metrics
**Procedure**:
1. Query Sustainalytics dataset for company
2. Retrieve E, S, G component scores
3. Calculate composite ESG score
4. Return structured ESG data object

### Commodity and Currency Methods

#### `getCommodityPrices(commodity)`
**Purpose**: Real-time and historical commodity pricing
**Parameters**:
- `commodity` (string): Commodity identifier ('CRUDE_OIL', 'GOLD', etc.)
**Returns**: Commodity price time series
**Risk Application**: Portfolio diversification and inflation hedge analysis

#### `getCurrencyExchangeRates(fromCurrency, toCurrency)`
**Purpose**: Foreign exchange rate data
**Parameters**:
- `fromCurrency` (string): Base currency code
- `toCurrency` (string): Quote currency code
**Returns**: Exchange rate time series
**Use Case**: Multi-currency portfolio risk calculations

---

## Risk Calculation Methods

### Volatility Calculations

#### `calculateVolatility(prices)`
**Purpose**: Historical volatility calculation using price series
**Mathematical Formula**: σ = √(Σ(r - μ)² / (n-1)) × √252
**Parameters**:
- `prices` (array): Historical price data
**Returns**: Annualized volatility (standard deviation)
**Procedure**:
1. Convert prices to log returns: r = ln(P_t / P_{t-1})
2. Calculate mean return: μ = Σr / n
3. Compute variance: σ² = Σ(r - μ)² / (n-1)
4. Annualize: multiply by √252 (trading days)
**Validation**: Minimum 30 data points required

#### `calculateBeta(stockPrices, marketPrices)`
**Purpose**: Systematic risk measurement relative to market
**Mathematical Formula**: β = Cov(R_stock, R_market) / Var(R_market)
**Parameters**:
- `stockPrices` (array): Individual stock price series
- `marketPrices` (array): Market index price series
**Returns**: Beta coefficient
**Procedure**:
1. Calculate returns for both stock and market
2. Compute covariance between return series
3. Calculate market variance
4. Divide covariance by market variance
**Interpretation**: β > 1 indicates higher volatility than market

### Risk-Adjusted Performance

#### `calculateSharpeRatio(prices, riskFreeRate)`
**Purpose**: Risk-adjusted return measurement
**Mathematical Formula**: SR = (R_portfolio - R_f) / σ_portfolio
**Parameters**:
- `prices` (array): Asset price series
- `riskFreeRate` (number): Risk-free rate (default: 2%)
**Returns**: Sharpe ratio
**Procedure**:
1. Calculate annualized return from price series
2. Subtract risk-free rate from return
3. Divide by annualized volatility
4. Higher values indicate better risk-adjusted performance

### Value at Risk (VaR) Methods

#### `calculateValueAtRisk(prices, confidenceLevel, timeHorizon)`
**Purpose**: Potential loss estimation at specified confidence level
**Method**: Historical simulation approach
**Parameters**:
- `prices` (array): Historical price data
- `confidenceLevel` (number): Confidence level (default: 0.95)
- `timeHorizon` (number): Time horizon in days (default: 1)
**Returns**: VaR estimate (negative value indicates loss)
**Procedure**:
1. Calculate historical returns from prices
2. Sort returns in ascending order
3. Find percentile corresponding to (1 - confidence level)
4. Scale by square root of time horizon
5. Return VaR estimate

#### `calculateExpectedShortfall(prices, confidenceLevel)`
**Purpose**: Expected loss beyond VaR threshold (Conditional VaR)
**Parameters**: Same as VaR calculation
**Returns**: Expected shortfall value
**Procedure**:
1. Calculate VaR at specified confidence level
2. Identify all returns worse than VaR threshold
3. Calculate average of tail returns
4. Return expected shortfall (more conservative than VaR)

### Portfolio Risk Methods

#### `calculatePortfolioRisk(holdings, correlationMatrix)`
**Purpose**: Portfolio-level risk calculation with correlations
**Mathematical Formula**: σ_p = √(w^T Σ w)
**Parameters**:
- `holdings` (array): Portfolio holdings with weights and volatilities
- `correlationMatrix` (2D array): Correlation matrix between assets
**Returns**: Portfolio volatility
**Procedure**:
1. Extract weights (w) and volatilities (σ) from holdings
2. Construct covariance matrix: Σ = D × C × D (where D = diag(σ), C = correlation matrix)
3. Calculate portfolio variance: σ²_p = w^T Σ w
4. Return portfolio volatility: σ_p = √σ²_p

---

## Data Flow Procedures

### 1. Data Collection Workflow

```
User Request → Parameter Validation → API Route Handler
     ↓
API Client Selection (FMP/Quandl) → Authentication Check
     ↓
Data Retrieval → Error Handling → Data Validation
     ↓
Data Transformation → Cache Storage → Response Formation
```

### 2. Risk Calculation Pipeline

```
Raw Price Data → Data Cleaning → Return Calculation
     ↓
Statistical Analysis → Risk Metric Computation → Validation
     ↓
Result Formatting → Error Checking → Response Delivery
```

### 3. Portfolio Analysis Procedure

```
Portfolio Input → Weight Validation → Individual Asset Analysis
     ↓
Correlation Matrix Calculation → Portfolio Risk Computation
     ↓
Diversification Analysis → Risk Attribution → Report Generation
```

---

## Error Handling Procedures

### API Error Management

#### Rate Limit Handling
**Procedure**:
1. Monitor request count per minute/day
2. Implement exponential backoff on rate limit errors
3. Queue requests when approaching limits
4. Provide user feedback on rate limit status

#### Authentication Errors
**Procedure**:
1. Validate API keys on startup
2. Refresh tokens if applicable
3. Provide clear error messages for invalid credentials
4. Log authentication failures for monitoring

#### Data Quality Validation
**Procedure**:
1. Check for null/undefined values
2. Validate data types and ranges
3. Verify time series continuity
4. Flag suspicious data patterns
5. Implement fallback data sources

### Calculation Error Handling

#### Insufficient Data Errors
**Triggers**:
- Less than minimum required data points
- Missing critical data fields
- Inconsistent time series

**Response**:
1. Return descriptive error message
2. Suggest alternative analysis methods
3. Provide data requirement specifications

#### Mathematical Errors
**Common Issues**:
- Division by zero in ratio calculations
- Negative values in logarithmic calculations
- Matrix singularity in portfolio calculations

**Mitigation**:
1. Input validation before calculations
2. Mathematical safeguards and bounds checking
3. Alternative calculation methods for edge cases
4. Clear error messaging with suggested remediation

---

## Performance Optimization Procedures

### Caching Strategy
1. **API Response Caching**: Cache API responses based on TTL settings
2. **Calculation Caching**: Store computed risk metrics with invalidation rules
3. **Memory Management**: Implement cache size limits and LRU eviction

### Request Optimization
1. **Batch Processing**: Group related API calls
2. **Parallel Execution**: Use Promise.all for independent requests
3. **Request Deduplication**: Avoid duplicate API calls within time windows

### Monitoring and Alerting
1. **Performance Metrics**: Track API response times and success rates
2. **Error Monitoring**: Log and alert on calculation failures
3. **Usage Analytics**: Monitor API usage patterns and costs