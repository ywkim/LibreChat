import { StructuredTool, ToolParams } from 'langchain/tools';
import { OpenAI } from 'langchain/llms/openai';
import { loadQARefineChain } from 'langchain/chains';
import { PlaywrightWebBaseLoader } from 'langchain/document_loaders/web/playwright';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { z } from 'zod';
import { Page, Frame } from 'playwright';

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
const processPage = async (page: Page | Frame): Promise<string> => {
    const pageText = await page.innerText('body');
    const title = await page.title();

    if (title) {
        return `# ${title}\n\n${pageText}`;
    }

    return pageText;
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
                gotoOptions: { waitUntil: 'networkidle' },
                evaluate: async (page: Page): Promise<string> => {
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
            throw new Error(String(e));
        }
    }
}

module.exports = WebQA;
