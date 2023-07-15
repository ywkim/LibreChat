const { Document } = require('langchain/document');
const { BaseDocumentLoader } = require('langchain/document_loaders/base');

class SerpAPILoader extends BaseDocumentLoader {
  constructor(apiKey, searchQuery, device = 'desktop', noCache = false, timeout = 0) {
    super();
    this.apiKey = apiKey;
    this.searchQuery = searchQuery;
    this.device = device;
    this.noCache = noCache;
    this.timeout = timeout;
  }

  async load() {
    const url = `https://serpapi.com/search?api_key=${this.apiKey}&q=${encodeURIComponent(
      this.searchQuery
    )}&device=${this.device}&no_cache=${this.noCache}&timeout=${this.timeout}`;

    try {
      const response = await fetch(url);
      const data = await response.json();

      if ('error' in data) {
        throw new Error(`Got error from SerpAPI: ${data['error']}`);
      }

      const responseTypes = [
        'answer_box',
        'sports_results',
        'shopping_results',
        'knowledge_graph',
        'organic_results'
      ];

      console.log('SerpAPI Response');
      console.log(data);

      const documents = [];

      for (const responseType of responseTypes) {
        if (responseType in data) {
          const output = data[responseType];
          const results = Array.isArray(output) ? output : [output];
          for (const result of results) {
            const pageContent = JSON.stringify(result);
            const metadata = { source: 'SerpAPI' };
            console.log(`responseType: ${responseType}`);
            console.log(pageContent);
            documents.push(new Document({ pageContent, metadata }));
          }
        }
      }

      return documents;
    } catch (error) {
      console.error(error);
      throw new Error('Failed to load search results from SerpAPI');
    }
  }
}

module.exports = SerpAPILoader;
