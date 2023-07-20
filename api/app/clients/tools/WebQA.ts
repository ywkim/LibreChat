import { StructuredTool, ToolParams } from 'langchain/tools';
import { OpenAI } from 'langchain/llms/openai';
import { loadQARefineChain } from 'langchain/chains';
import { Document } from 'langchain/document';
import { BaseDocumentLoader } from 'langchain/document_loaders/base';
import type { DocumentLoader } from 'langchain/document_loaders/base';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { z } from 'zod';
import { LaunchOptions, Page, Browser, Response, Frame } from 'playwright';

export type PlaywrightGotoOptions = {
    referer?: string;
    timeout?: number;
    waitUntil?: "load" | "domcontentloaded" | "networkidle" | "commit";
};

export type PlaywrightEvaluate = (
    page: Page,
    browser: Browser,
    response: Response | null
) => Promise<string>;

export type PlaywrightWebBaseLoaderOptions = {
    launchOptions?: LaunchOptions;
    gotoOptions?: PlaywrightGotoOptions;
    evaluate?: PlaywrightEvaluate;
};

export class PlaywrightWebBaseLoader
    extends BaseDocumentLoader
    implements DocumentLoader {
    options: PlaywrightWebBaseLoaderOptions | undefined;

    constructor(
        public webPath: string,
        options?: PlaywrightWebBaseLoaderOptions
    ) {
        super();
        this.options = options ?? undefined;
    }

    static async _scrape(
        url: string,
        options?: PlaywrightWebBaseLoaderOptions
    ): Promise<string> {
        const { chromium } = await PlaywrightWebBaseLoader.imports();

        const browser = await chromium.launch({
            headless: true,
            ...options?.launchOptions,
        });
        const page = await browser.newPage();

        const response = await page.goto(url, {
            timeout: 180000,
            waitUntil: "domcontentloaded",
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
        chromium: typeof import("playwright").chromium;
    }> {
        try {
            const { chromium } = await import("playwright");

            return { chromium };
        } catch (e) {
            console.error(e);
            throw new Error(
                "Please install playwright as a dependency with, e.g. `yarn add playwright`"
            );
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
    llm: OpenAI;
}

export default class WebQA extends StructuredTool {
    name = 'ask-webpage';
    description = 'Use this when you need to answer questions about specific webpages';
    schema = z.object({
        question: z.string().describe('should be a question on response content'),
        url: z.string().describe('should be a string'),
    });

    private llm: OpenAI;

    constructor({ llm }: WebQAArgs) {
        super();
        this.llm = llm;
    }

    async _call({ question, url }: WebQASchema): Promise<string> {
        console.log(`WebQA question: ${question}, url: ${url}, llm: ${this.llm.modelName}`);
        try {
            const loader = new PlaywrightWebBaseLoader(url, {
                gotoOptions: { waitUntil: 'load' },
                evaluate: async (page: Page, browser: Browser, response: Response | null): Promise<string> => {
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
                }
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
            throw new Error(String(e));
        }
    }
}

module.exports = WebQA;
