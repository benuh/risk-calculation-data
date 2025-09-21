# Risk Calculation Data Platform - Setup Guide

## Quick Start Guide

This guide will walk you through setting up the Risk Calculation Data Platform from scratch, including API key acquisition, environment configuration, and running your first risk calculations.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [API Key Setup](#api-key-setup)
3. [Project Installation](#project-installation)
4. [Environment Configuration](#environment-configuration)
5. [Running the Application](#running-the-application)
6. [Testing the Setup](#testing-the-setup)
7. [Common Issues and Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Node.js**: Version 14.0 or higher
- **npm**: Version 6.0 or higher (comes with Node.js)
- **Git**: For cloning the repository
- **Text Editor**: VS Code, Sublime Text, or your preferred editor

### Verify Prerequisites
```bash
# Check Node.js version
node --version
# Should output: v14.x.x or higher

# Check npm version
npm --version
# Should output: 6.x.x or higher

# Check Git installation
git --version
# Should output: git version 2.x.x
```

---

## API Key Setup

### 1. Financial Modeling Prep (FMP) API Key

#### Step 1: Create FMP Account
1. Visit [Financial Modeling Prep](https://financialmodelingprep.com/)
2. Click "Sign Up" or "Get Started"
3. Choose your subscription plan:
   - **Free Plan**: 250 API calls/day
   - **Starter Plan**: $14/month - 10,000 calls/day
   - **Professional Plan**: $29/month - 100,000 calls/day

#### Step 2: Obtain API Key
1. Log into your FMP dashboard
2. Navigate to "Dashboard" → "API Key"
3. Copy your API key (format: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)
4. **Important**: Keep this key secure and never commit it to version control

#### API Key Validation
Test your FMP API key:
```bash
curl "https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=YOUR_FMP_API_KEY"
```

### 2. Quandl API Key

#### Step 1: Create Quandl Account
1. Visit [Quandl](https://www.quandl.com/)
2. Click "Sign Up" in the top right
3. Complete the registration process
4. **Note**: Quandl offers both free and premium datasets

#### Step 2: Get API Key
1. Log into your Quandl account
2. Go to "Account Settings" → "API Key"
3. Copy your API key (format: `xxxxxxxxxxxxxxxxxxxx`)

#### API Key Validation
Test your Quandl API key:
```bash
curl "https://www.quandl.com/api/v3/datasets/FRED/GDP.json?api_key=YOUR_QUANDL_API_KEY"
```

---

## Project Installation

### Option 1: Clone from Repository
```bash
# Clone the repository
git clone <repository-url>
cd risk-calculation-data
```

### Option 2: Download and Extract
1. Download the project as a ZIP file
2. Extract to your desired directory
3. Navigate to the project folder

### Install Dependencies
```bash
# Install all required packages
npm install

# This will install:
# - express (web framework)
# - axios (HTTP client)
# - dotenv (environment variables)
# - cors (cross-origin resource sharing)
# - And development dependencies (jest, nodemon, eslint, prettier)
```

### Verify Installation
```bash
# Check that node_modules directory was created
ls -la
# Should see node_modules/ directory

# Verify package installation
npm list --depth=0
# Should show all installed packages
```

---

## Environment Configuration

### 1. Create Environment File
```bash
# Copy the example environment file
cp .env.example .env
```

### 2. Configure Environment Variables

Edit the `.env` file with your actual API keys:

```env
# Financial Modeling Prep API Configuration
FMP_API_KEY=your_actual_fmp_api_key_here
FMP_BASE_URL=https://financialmodelingprep.com/api/v3

# Quandl API Configuration
QUANDL_API_KEY=your_actual_quandl_api_key_here
QUANDL_BASE_URL=https://www.quandl.com/api/v3

# Server Configuration
PORT=3000
NODE_ENV=development

# Cache Configuration
CACHE_TTL=300

# Optional: Database Configuration (for future use)
# DATABASE_URL=your_database_url_here
```

### 3. Environment Variable Validation

The application will validate your configuration on startup. Required variables:
- `FMP_API_KEY`: Must be a valid FMP API key
- `QUANDL_API_KEY`: Must be a valid Quandl API key

---

## Running the Application

### Development Mode (Recommended for Testing)
```bash
# Start with auto-reload on file changes
npm run dev

# Expected output:
# Risk Calculation Data Server running on port 3000
# Environment: development
# API endpoints available at http://localhost:3000/api/
```

### Production Mode
```bash
# Start in production mode
npm start

# Or with environment variable
NODE_ENV=production npm start
```

### Background Process
```bash
# Run as background process (Linux/Mac)
nohup npm start > app.log 2>&1 &

# Check if running
ps aux | grep node
```

---

## Testing the Setup

### 1. Health Check
```bash
# Test basic server functionality
curl http://localhost:3000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "1.0.0"
}
```

### 2. API Key Validation Test
```bash
# Test FMP integration
curl "http://localhost:3000/api/risk/AAPL"

# Expected response (partial):
{
  "symbol": "AAPL",
  "riskMetrics": {
    "calculationDate": "2024-01-15T10:30:00.000Z",
    "timeHorizon": "1Y"
  },
  "priceData": [...],
  "calculatedAt": "2024-01-15T10:30:00.000Z"
}
```

### 3. Economic Data Test
```bash
# Test Quandl integration
curl "http://localhost:3000/api/economic-indicators?indicator=GDP"

# Expected response:
{
  "indicator": "GDP",
  "country": "US",
  "fmpData": [...],
  "quandlData": [...],
  "retrievedAt": "2024-01-15T10:30:00.000Z"
}
```

### 4. Treasury Yields Test
```bash
# Test treasury data
curl "http://localhost:3000/api/treasury-yields"

# Expected response:
{
  "treasuryYields": [...],
  "retrievedAt": "2024-01-15T10:30:00.000Z"
}
```

---

## Development Workflow

### 1. Code Quality Tools
```bash
# Run linting
npm run lint

# Fix linting issues automatically
npm run lint -- --fix

# Format code with Prettier
npm run format
```

### 2. Testing
```bash
# Run test suite (when implemented)
npm test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

### 3. Debugging

#### Enable Debug Logging
```bash
# Set debug environment variable
DEBUG=risk-calc:* npm run dev
```

#### VS Code Debug Configuration
Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Launch Risk Calc Server",
      "type": "node",
      "request": "launch",
      "program": "${workspaceFolder}/src/index.js",
      "env": {
        "NODE_ENV": "development"
      },
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "FMP_API_KEY is required" Error
**Problem**: Missing or invalid FMP API key
**Solution**:
```bash
# Check .env file exists and has correct key
cat .env | grep FMP_API_KEY

# Test API key manually
curl "https://financialmodelingprep.com/api/v3/quote/AAPL?apikey=YOUR_KEY"
```

#### 2. "QUANDL_API_KEY is required" Error
**Problem**: Missing or invalid Quandl API key
**Solution**:
```bash
# Verify Quandl key in .env
cat .env | grep QUANDL_API_KEY

# Test Quandl API access
curl "https://www.quandl.com/api/v3/datasets/FRED/GDP.json?api_key=YOUR_KEY"
```

#### 3. Port Already in Use Error
**Problem**: Port 3000 is already occupied
**Solutions**:
```bash
# Option 1: Kill process using port 3000
lsof -ti:3000 | xargs kill

# Option 2: Use different port
PORT=3001 npm run dev

# Option 3: Change port in .env file
echo "PORT=3001" >> .env
```

#### 4. Module Not Found Errors
**Problem**: Dependencies not installed correctly
**Solution**:
```bash
# Clear npm cache and reinstall
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### 5. API Rate Limit Errors
**Problem**: Exceeding API call limits
**Solutions**:
- Check your API plan limits
- Implement request throttling
- Use caching to reduce API calls
- Upgrade to higher tier plan

#### 6. CORS Errors in Browser
**Problem**: Cross-origin requests blocked
**Solution**: The server includes CORS middleware, but if issues persist:
```javascript
// Modify src/index.js if needed
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001'],
  credentials: true
}));
```

### Performance Optimization

#### 1. Enable Caching
```env
# In .env file, adjust cache settings
CACHE_TTL=600  # 10 minutes
```

#### 2. Monitor API Usage
```bash
# Track API call counts
grep "API call" app.log | wc -l

# Monitor response times
tail -f app.log | grep "response time"
```

#### 3. Database Integration (Future)
For production deployment, consider adding database support for:
- Caching API responses
- Storing calculated risk metrics
- User management and preferences

---

## Next Steps

### 1. Explore the API
- Review the [API Documentation](./API_METHODS.md)
- Test different endpoints with various parameters
- Understand the risk calculation methods

### 2. Customize for Your Needs
- Modify risk calculation parameters
- Add new data sources
- Implement custom risk models

### 3. Production Deployment
- Set up production environment
- Configure database and caching
- Implement monitoring and logging
- Set up CI/CD pipeline

### 4. Integration Options
- Build a web frontend
- Create Excel/Python integrations
- Develop mobile applications
- Set up automated reporting

---

## Support and Resources

### Documentation
- [Main README](../README.md)
- [API Methods Documentation](./API_METHODS.md)
- [Financial Modeling Prep API Docs](https://financialmodelingprep.com/developer/docs)
- [Quandl API Documentation](https://docs.quandl.com/)

### Getting Help
1. Check this troubleshooting guide
2. Review the API documentation
3. Search existing issues in the repository
4. Create a new issue with detailed error information

### Contributing
If you encounter issues or have improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your changes

---

## Security Best Practices

### API Key Security
- Never commit `.env` files to version control
- Use different API keys for development and production
- Regularly rotate API keys
- Monitor API key usage for unusual activity

### Environment Security
- Restrict access to production servers
- Use HTTPS in production
- Implement proper authentication for admin endpoints
- Regular security updates for dependencies

### Data Protection
- Implement rate limiting
- Validate all input parameters
- Sanitize data before processing
- Log security events for monitoring