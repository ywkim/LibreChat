const { StructuredTool, ToolParams } = require('langchain/tools');
const { OpenAI } = require('langchain/llms/openai');
const { RetrievalQAChain, loadQARefineChain } = require('langchain/chains');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { OpenAIEmbeddings } = require('langchain/embeddings/openai');
const { z } = require('zod');
const SerpAPILoader = require('./SerpAPILoader');

class SearchQA extends StructuredTool {
  constructor({ llm, embeddings, apiKey }) {
    super();
    this.name = 'search-and-ask';
    this.description = 'Useful for when you need to answer questions about current events';
    this.schema = z.object({
      question: z.string().describe('should be a question on Google search results'),
      query: z
        .string()
        .describe(
          'should be a Google search query. You can use anything that you would use in a regular Google search.'
        )
    });
    this.llm = llm;
    this.embeddings = embeddings;
    this.apiKey = apiKey;
  }

  async _call({ question, query }) {
    console.log('SearchQA Tool');
    console.log(`Question: ${question}`);
    console.log(`Query: ${query}`);
    try {
      const loader = new SerpAPILoader(this.apiKey, query);

      // SerpAPILoader already creates documents in the right size.
      // So there is no need to use loadAndSplit.
      const docs = await loader.load();

      const vectorStore = await MemoryVectorStore.fromDocuments(docs, this.embeddings);
      const retriever = vectorStore.asRetriever();

      const retreived = await retriever.getRelevantDocuments(question);
      console.log(`< ======= relevant docs ======= >`);
      console.log(retreived);

      const chain = RetrievalQAChain.fromLLM(this.llm, retriever);
      const answer = await chain.call({ query: question });
      console.log('Search ANSWER:');
      console.log(answer);
      return answer.text;
    } catch (e) {
      console.error(e);
      throw new Error(e.message);
    }
  }
}

module.exports = SearchQA;
