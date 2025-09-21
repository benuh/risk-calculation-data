const axios = require('axios');
require('dotenv').config();

class FMPClient {
  constructor() {
    this.apiKey = process.env.FMP_API_KEY;
    this.baseUrl = process.env.FMP_BASE_URL || 'https://financialmodelingprep.com/api/v3';

    if (!this.apiKey) {
      throw new Error('FMP_API_KEY is required in environment variables');
    }

    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 10000,
      params: {
        apikey: this.apiKey
      }
    });
  }

  async getEarningsTranscripts(symbol, year = null, quarter = null) {
    try {
      let endpoint = `/earning_call_transcript/${symbol}`;
      const params = {};

      if (year) params.year = year;
      if (quarter) params.quarter = quarter;

      const response = await this.client.get(endpoint, { params });
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch earnings transcripts: ${error.message}`);
    }
  }

  async getEconomicIndicators(indicator = 'GDP') {
    try {
      const response = await this.client.get(`/economic?name=${indicator}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch economic indicators: ${error.message}`);
    }
  }

  async getTreasuryYields() {
    try {
      const response = await this.client.get('/treasury');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch treasury yields: ${error.message}`);
    }
  }

  async getRiskPremium(symbol) {
    try {
      const response = await this.client.get(`/risk-premium/${symbol}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch risk premium for ${symbol}: ${error.message}`);
    }
  }

  async getStockPrice(symbol) {
    try {
      const response = await this.client.get(`/quote-short/${symbol}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch stock price for ${symbol}: ${error.message}`);
    }
  }

  async getCompanyProfile(symbol) {
    try {
      const response = await this.client.get(`/profile/${symbol}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch company profile for ${symbol}: ${error.message}`);
    }
  }

  async getFinancialStatements(symbol, type = 'income-statement', period = 'annual') {
    try {
      const response = await this.client.get(`/${type}/${symbol}?period=${period}`);
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch ${type} for ${symbol}: ${error.message}`);
    }
  }

  async getMarketRiskPremium() {
    try {
      const response = await this.client.get('/market-risk-premium');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to fetch market risk premium: ${error.message}`);
    }
  }
}

module.exports = FMPClient;