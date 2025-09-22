const axios = require('axios');
require('dotenv').config();

class QuandlClient {
  constructor() {
    this.apiKey = process.env.QUANDL_API_KEY;
    this.baseUrl = process.env.QUANDL_BASE_URL || 'https://www.quandl.com/api/v3';

    if (!this.apiKey) {
      throw new Error('QUANDL_API_KEY is required in environment variables');
    }

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 10000,
      params: {
        api_key: this.apiKey
      }
    });
  }

  async getMacroeconomicData(dataset, code, startDate = null, endDate = null) {
    try {
      const params = {};
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;

      const response = await this.client.get(`/datasets/${dataset}/${code}/data.json`, { params });
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch macroeconomic data: ${error.message}`);
    }
  }

  async getESGScores(symbol) {
    try {
      const response = await this.client.get(`/datasets/SUSTAINALYTICS/${symbol}/data.json`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch ESG scores for ${symbol}: ${error.message}`);
    }
  }

  async getCentralBankForecasts(country = 'US', indicator = 'GDP') {
    try {
      const dataset = `OECD/${country}_${indicator}_FORECAST`;
      const response = await this.client.get(`/datasets/${dataset}/data.json`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch central bank forecasts: ${error.message}`);
    }
  }

  async getGovernmentData(dataset, code) {
    try {
      const response = await this.client.get(`/datasets/${dataset}/${code}/data.json`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch government data: ${error.message}`);
    }
  }

  async getInflationData(country = 'USA') {
    try {
      const dataset = `RATEINF/${country}_INFL_M`;
      const response = await this.client.get(`/datasets/${dataset}/data.json`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch inflation data for ${country}: ${error.message}`);
    }
  }

  async getUnemploymentData(country = 'USA') {
    try {
      const dataset = 'FRED/UNRATE';
      const response = await this.client.get(`/datasets/${dataset}/data.json`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch unemployment data: ${error.message}`);
    }
  }

  async getCommodityPrices(commodity = 'CRUDE_OIL') {
    try {
      const dataset = `CHRIS/CME_${commodity}1`;
      const response = await this.client.get(`/datasets/${dataset}/data.json`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch commodity prices for ${commodity}: ${error.message}`);
    }
  }

  async getCurrencyExchangeRates(fromCurrency = 'USD', toCurrency = 'EUR') {
    try {
      const dataset = `CURRFX/${fromCurrency}${toCurrency}`;
      const response = await this.client.get(`/datasets/${dataset}/data.json`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch exchange rates ${fromCurrency}/${toCurrency}: ${error.message}`);
    }
  }

  async searchDatasets(query, limit = 10) {
    try {
      const response = await this.client.get('/datasets.json', {
        params: {
          query: query,
          per_page: limit
        }
      });
      return response.data;
    } catch (error) {
      throw new Error(`Failed to search datasets: ${error.message}`);
    }
  }
}

module.exports = QuandlClient;