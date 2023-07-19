const { StructuredTool, ToolParams } = require('langchain/tools');
const { OpenAI } = require('langchain/llms/openai');
const { loadQARefineChain } = require('langchain/chains');
const { CheerioWebBaseLoader } = require('langchain/document_loaders/web/cheerio');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { OpenAIEmbeddings } = require('langchain/embeddings/openai');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { z } = require('zod');

const getMaxTokens = (modelName) => {
  if (modelName.startsWith('gpt-3.5-turbo-16k')) {
    return 16384;
  }
  if (modelName.startsWith('gpt-4-32k')) {
    return 32768;
  }
  if (modelName.startsWith('gpt-4')) {
    return 8192;
  }
  return 4096;
};

class WebQA extends StructuredTool {
  name = 'ask-webpage';
  description = 'Use this when you need to answer questions about specific webpages';
  schema = z.object({
    question: z.string().describe('should be a question on response content'),
    urls: z.array(z.string()).describe('should be a list of strings')
  });

  constructor({ embeddings, llm }) {
    super();
    this.embeddings = embeddings;
    this.llm = llm;
  }

  async _call({ question, urls }) {
    console.log(`WebQA question: ${question}, url: ${urls}, llm: ${this.llm.modelName}`);
    try {
      const loader = new CheerioWebBaseLoader(urls[0]);

      const maxToken = getMaxTokens(this.llm.modelName);

      const chunkSize = Math.floor(maxToken * 0.9);
      const chunkOverlap = Math.floor(chunkSize * 0.2);
      const lengthFunction = this.llm.getNumTokens;

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
        lengthFunction
      });

      // Using loadAndSplit cuts the document in the wrong place, and as a result similarity search doesn't seem to work either.
      // So instead of using MemoryVectorStore.fromDocuments and RetreivalQAChain.fromLLM, we do refine document QA.
      const docs = await loader.loadAndSplit(splitter);

      console.log(`< ======= docs (total: ${docs.length}) ======= >`);
      docs.forEach((doc, i) => {
        console.log(`\n[Document ${i}]\n`);
        console.log(doc);
      });

      const chain = loadQARefineChain(this.llm);
      const answer = await chain.call({
        input_documents: docs,
        question
      });

      return answer.output_text;
    } catch (e) {
      console.error(e);
      throw new Error(e.message);
    }
  }
}

module.exports = WebQA;
