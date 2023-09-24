import { StructuredTool, ToolParams } from 'langchain/tools';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { loadQARefineChain, RetrievalQAChain } from 'langchain/chains';
import { Document } from 'langchain/document';
import { BaseDocumentLoader } from 'langchain/document_loaders/base';
import type { DocumentLoader } from 'langchain/document_loaders/base';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { z } from 'zod';
import { LaunchOptions, Page, Browser, Response, Frame, Locator } from 'playwright';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

export type PlaywrightGotoOptions = {
  referer?: string;
  timeout?: number;
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle' | 'commit';
};

export type PlaywrightEvaluate = (
  page: Page,
  browser: Browser,
  response: Response | null,
) => Promise<string>;

export type PlaywrightWebBaseLoaderOptions = {
  launchOptions?: LaunchOptions;
  gotoOptions?: PlaywrightGotoOptions;
  evaluate?: PlaywrightEvaluate;
};

export class PlaywrightWebBaseLoader extends BaseDocumentLoader implements DocumentLoader {
  options: PlaywrightWebBaseLoaderOptions | undefined;

  constructor(public webPath: string, options?: PlaywrightWebBaseLoaderOptions) {
    super();
    this.options = options ?? undefined;
  }

  static async _scrape(url: string, options?: PlaywrightWebBaseLoaderOptions): Promise<string> {
    const { chromium } = await PlaywrightWebBaseLoader.imports();

    const browser = await chromium.launch({
      headless: true,
      ...options?.launchOptions,
    });
    const page = await browser.newPage();

    const response = await page.goto(url, {
      timeout: 180000,
      waitUntil: 'domcontentloaded',
      ...options?.gotoOptions,
    });
    const bodyHTML = options?.evaluate
      ? await options?.evaluate(page, browser, response)
      : await page.content();

    await browser.close();

    return bodyHTML;
  }

  async scrape(): Promise<string> {
    return PlaywrightWebBaseLoader._scrape(this.webPath, this.options);
  }

  async load(): Promise<Document[]> {
    const text = await this.scrape();

    const metadata = { source: this.webPath };
    return [new Document({ pageContent: text, metadata })];
  }

  static async imports(): Promise<{
    chromium: typeof import('playwright').chromium;
  }> {
    try {
      const { chromium } = await import('playwright');

      return { chromium };
    } catch (e) {
      console.error(e);
      throw new Error('Please install playwright as a dependency with, e.g. `yarn add playwright`');
    }
  }
}

const getMaxTokens = (modelName: string): number => {
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

// If it can be created as a separate Document, would it be better?
const evaluateFrameContent = async (page: Page | Frame): Promise<string> => {
  const scripts: Locator = page.locator('script');
  await scripts.evaluateAll((scripts) => scripts.forEach((script) => script.remove()));

  const body = page.locator('body');
  const bodyText = await body.innerText();
  const title = await page.title();

  if (title) {
    return `# ${title}\n\n${bodyText}`;
  }

  return bodyText;
};

interface WebQASchema {
  question: string;
  url: string;
}

interface WebQAArgs extends ToolParams {
  embeddings: OpenAIEmbeddings;
  llm: ChatOpenAI;
}

export default class WebQA extends StructuredTool {
  name = 'ask-webpage';
  description = 'Use this when you need to answer questions about specific webpages';
  schema = z.object({
    question: z.string().describe('should be a question on response content'),
    url: z.string().describe('should be a string'),
  });

  private llm: ChatOpenAI;
  private embeddings: OpenAIEmbeddings;

  constructor({ llm, embeddings }: WebQAArgs) {
    super();
    this.llm = llm;
    this.embeddings = embeddings;
  }

  async _call({ question, url }: WebQASchema): Promise<string> {
    console.log(`WebQA question: ${question}, url: ${url}, llm: ${this.llm.modelName}`);
    try {
      const loader = new PlaywrightWebBaseLoader(url, {
        gotoOptions: { waitUntil: 'load' },
        evaluate: async (
          page: Page,
          browser: Browser,
          response: Response | null,
        ): Promise<string> => {
          let contentType = 'text/html';
          if (response != null) {
            contentType = response.headers()['content-type'];
          }
          if (contentType.includes('text/html')) {
            const frames = page.frames();
            const contents = await Promise.all(frames.map(evaluateFrameContent));
            return contents.join('\n\n');
          }
          return await response!.text();
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

      const docs = await loader.loadAndSplit(splitter);

      const combineDocumentsChain = await loadQARefineChain(this.llm);

      // If the number of documents is greater than 4, we use RetrievalQAChain with MemoryVectorStore.
      // This is more efficient for large documents as it allows us to perform similarity search on the documents in memory.

      if (docs.length <= 4) {
        console.log(`< ======= docs (total: ${docs.length}) ======= >`);
        docs.forEach((doc, i) => {
          console.log(`\n[Document ${i}]\n`);
          console.log(doc);
        });

        // If there are 4 or fewer documents, use the original method.
        const answer = await combineDocumentsChain.call({
          input_documents: docs,
          question,
        });
        return answer.output_text;
      } else {
        // If there are more than 4 documents, use the RetrievalQAChain.
        const vectorStore = await MemoryVectorStore.fromDocuments(docs, this.embeddings);
        const retriever = vectorStore.asRetriever();

        const relevantDocs = await retriever.getRelevantDocuments(question);
        console.log(
          `< ======= docs (relevant: ${relevantDocs.length}, total: ${docs.length}) ======= >`,
        );
        relevantDocs.forEach((doc, i) => {
          console.log(`\n[Relevant Document ${i}]\n`);
          console.log(doc);
        });

        const chain = new RetrievalQAChain({
          combineDocumentsChain: combineDocumentsChain,
          retriever: retriever,
        });
        const result = await chain.call({ query: question });
        return result.text;
      }
    } catch (e) {
      console.error(e);
      throw new Error(String(e));
    }
  }
}

module.exports = WebQA;
