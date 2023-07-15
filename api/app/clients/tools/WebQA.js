const { StructuredTool, ToolParams } = require('langchain/tools');
const { OpenAI } = require('langchain/llms/openai');
const { RetrievalQAChain, loadQARefineChain } = require('langchain/chains');
const { CheerioWebBaseLoader } = require('langchain/document_loaders/web/cheerio');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { OpenAIEmbeddings } = require('langchain/embeddings/openai');
const { z } = require('zod');

class WebQA extends StructuredTool {
  constructor({ embeddings, llm }) {
    super();
    this.embeddings = embeddings;
    this.llm = llm;
  }

  name = 'ask-webpage';
  description = 'Use this when you need to answer questions about specific webpages';
  schema = z.object({
    question: z.string().describe('should be a question on response content'),
    urls: z.array(z.string()).describe('should be a list of strings')
  });

  async _call({ question, urls }) {
    console.log(`WebQA question: ${question}, url: ${urls}`);
    try {
      const loader = new CheerioWebBaseLoader(urls[0]);

      // Using loadAndSplit cuts the document in the wrong place, and as a result similarity search doesn't seem to work either.
      // So instead of using MemoryVectorStore.fromDocuments and RetreivalQAChain.fromLLM, we do refine document QA.
      const docs = await loader.loadAndSplit();

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
