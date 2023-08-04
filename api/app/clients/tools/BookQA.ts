import { StructuredTool, ToolParams } from 'langchain/tools';
import { ChatOpenAI } from "langchain/chat_models/openai";
import { RetrievalQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { z } from 'zod';
import { PineconeClient } from '@pinecone-database/pinecone';

type VectorOperationsApi = ReturnType<PineconeClient["Index"]>;

interface BookQASchema {
    question: string;
    book_id: string;
}

interface BookQAArgs extends ToolParams {
    embeddings: OpenAIEmbeddings;
    llm: ChatOpenAI;
    pineconeIndex: VectorOperationsApi;
}

export default class BookQA extends StructuredTool {
    name = 'ask-book';
    description = 'Use this when you need to answer questions about specific book or document.';
    schema = z.object({
        question: z.string(),
        book_id: z.string(),
    });

    private llm: ChatOpenAI;
    private embeddings: OpenAIEmbeddings;
    private pineconeIndex: VectorOperationsApi;

    constructor({ llm, embeddings, pineconeIndex }: BookQAArgs) {
        super();
        this.llm = llm;
        this.embeddings = embeddings;
        this.pineconeIndex = pineconeIndex;
    }

    async _call({ question, book_id }: BookQASchema): Promise<string> {
        console.log(`BookQA question: ${question}, book_id: ${book_id}, llm: ${this.llm.modelName}`);
        try {
            const stats = await this.pineconeIndex.describeIndexStats({ describeIndexStatsRequest: {} });
            const namespaces = stats.namespaces;
            if (!namespaces) {
                throw new Error(`Failed to retrieve namespaces`);
            }
            if (!(book_id in namespaces)) {
                throw new Error(`Invalid book_id: ${book_id}`);
            }
            if (namespaces[book_id].vectorCount === 0) {
                throw new Error(`Namespace ${book_id} is empty`);
            }

            const vectorStore = await PineconeStore.fromExistingIndex(this.embeddings, { pineconeIndex: this.pineconeIndex, namespace: book_id });
            const retriever = vectorStore.asRetriever();

            const matchedDocs = await retriever.getRelevantDocuments(question);
            console.log(`Matched docs: ${matchedDocs.length}`);
            matchedDocs.forEach((doc, i) => {
                console.log(`\n[Document ${i}]\n`);
                console.log(doc);
            });

            const qa = RetrievalQAChain.fromLLM(this.llm, retriever);
            const answer = await qa.call({ query: question });
            return answer.text;
        } catch (e) {
            console.error(e);
            throw new Error(String(e));
        }
    }
}

module.exports = BookQA;
