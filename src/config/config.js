require('dotenv').config();

const config = {
  app: {
    port: process.env.PORT || 3000,
    env: process.env.NODE_ENV || 'development'
  },

  apis: {
    fmp: {
      apiKey: process.env.FMP_API_KEY,
      baseUrl: process.env.FMP_BASE_URL || 'https://financialmodelingprep.com/api/v3',
      timeout: 10000,
      rateLimit: {
        requestsPerMinute: 250,
        requestsPerDay: 5000
      }
    },

    quandl: {
      apiKey: process.env.QUANDL_API_KEY,
      baseUrl: process.env.QUANDL_BASE_URL || 'https://www.quandl.com/api/v3',
      timeout: 10000,
      rateLimit: {
        requestsPerMinute: 300,
        requestsPerDay: 50000
      }
    }
  },

  cache: {
    ttl: parseInt(process.env.CACHE_TTL) || 300, // 5 minutes default
    maxSize: 1000,
    enabled: process.env.NODE_ENV !== 'test'
  },

  database: {
    url: process.env.DATABASE_URL,
    options: {
      useNewUrlParser: true,
      useUnifiedTopology: true
    }
  },

  riskCalculation: {
    defaultRiskFreeRate: 0.02,
    defaultConfidenceLevel: 0.95,
    defaultTimeHorizon: 252, // trading days in a year
    minimumDataPoints: 30
  },

  logging: {
    level: process.env.LOG_LEVEL || 'info',
    format: process.env.NODE_ENV === 'production' ? 'json' : 'simple'
  },

  security: {
    cors: {
      origin: process.env.CORS_ORIGIN || '*',
      credentials: true
    },
    rateLimit: {
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100 // limit each IP to 100 requests per windowMs
    }
  }
};

function validateConfig() {
  const errors = [];

  if (!config.apis.fmp.apiKey) {
    errors.push('FMP_API_KEY is required');
  }

  if (!config.apis.quandl.apiKey) {
    errors.push('QUANDL_API_KEY is required');
  }

  if (errors.length > 0) {
    throw new Error(`Configuration validation failed: ${errors.join(', ')}`);
  }

  return true;
}

function getConfig() {
  validateConfig();
  return config;
}

module.exports = {
  config,
  getConfig,
  validateConfig
};