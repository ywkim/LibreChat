import { StructuredTool, ToolParams } from 'langchain/tools';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { RetrievalQAChain } from 'langchain/chains';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { z } from 'zod';
import SerpAPILoader from './SerpAPILoader';

interface SearchQASchema {
  question: string;
  query: string;
}

interface SearchQAArgs extends ToolParams {
  embeddings: OpenAIEmbeddings;
  llm: ChatOpenAI;
  apiKey: string;
}

export default class SearchQA extends StructuredTool {
  name = 'search-and-ask';
  description = 'Useful for when you need to answer questions about current events';
  schema = z.object({
    question: z.string().describe('should be a question on Google search results'),
    query: z
      .string()
      .describe(
        'should be a Google search query. You can use anything that you would use in a regular Google search.',
      ),
  });

  private llm: ChatOpenAI;
  private embeddings: OpenAIEmbeddings;
  private apiKey: string;

  constructor({ llm, embeddings, apiKey }: SearchQAArgs) {
    super();
    this.llm = llm;
    this.embeddings = embeddings;
    this.apiKey = apiKey;
  }

  async _call({ question, query }: SearchQASchema): Promise<string> {
    console.log('SearchQA Tool');
    console.log(`Question: ${question}`);
    console.log(`Query: ${query}`);
    try {
      const loader = new SerpAPILoader({ apiKey: this.apiKey, searchQuery: query });

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
      throw new Error(String(e));
    }
  }
}

module.exports = SearchQA;
