const { StructuredTool, ToolParams } = require('langchain/tools');
const { OpenAI } = require('langchain/llms/openai');
const { loadQARefineChain } = require('langchain/chains');
const { PlaywrightWebBaseLoader } = require('langchain/document_loaders/web/playwright');
const { OpenAIEmbeddings } = require('langchain/embeddings/openai');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { z } = require('zod');
const { Page, Frame } = require('playwright');

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

const processPage = async (page) => {
  const pageText = await page.innerText('body');
  const title = await page.title();

  // If it can be created as a separate Document, would it be better?
  return `# ${title}\n\n${pageText}`;
};

class WebQA extends StructuredTool {
  name = 'ask-webpage';
  description = 'Use this when you need to answer questions about specific webpages';
  schema = z.object({
    question: z.string().describe('should be a question on response content'),
    url: z.string().describe('should be a string'),
  });

  constructor({ embeddings, llm }) {
    super();
    this.embeddings = embeddings;
    this.llm = llm;
  }

  async _call({ question, url }) {
    console.log(`WebQA question: ${question}, url: ${url}, llm: ${this.llm.modelName}`);
    try {
      const loader = new PlaywrightWebBaseLoader(url, {
        gotoOptions: { waitUntil: 'networkidle' },
        evaluate: async (page) => {
          const pageText = await processPage(page);
          const iframeElement = await page.$('iframe');
          if (iframeElement) {
            const iframePage = await iframeElement.contentFrame();
            if (iframePage) {
              const iframeText = await processPage(iframePage);
              return (pageText ?? '') + '\n\n' + (iframeText ?? '');
            }
          }
          return pageText ?? '';
        },
      });

      const maxToken = getMaxTokens(this.llm.modelName);

      const chunkSize = Math.floor(maxToken * 0.9);
      const chunkOverlap = Math.floor(chunkSize * 0.2);
      const lengthFunction = this.llm.getNumTokens;

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
        lengthFunction,
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
        question,
      });

      return answer.output_text;
    } catch (e) {
      console.error(e);
      throw new Error(e.message);
    }
  }
}

module.exports = WebQA;
