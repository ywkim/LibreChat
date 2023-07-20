import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { OpenAI } from "langchain/llms";
import WebQA from "./api/app/clients/tools/WebQA";
import * as fs from 'fs';
import * as util from 'util';
import * as dotenv from 'dotenv';
import * as yargs from 'yargs';

dotenv.config();

const DEFAULT_CONFIG = {
    settings: {
        chat_model: "gpt-4",
        temperature: "0",
    },
};

const readFile = util.promisify(fs.readFile);

async function loadConfig(configFile: string) {
    let config = DEFAULT_CONFIG;
    if (fs.existsSync(configFile)) {
        const fileData = await readFile(configFile, 'utf-8');
        config = JSON.parse(fileData);
    }
    return config;
}

function loadTools(config: any) {
    const llm = new OpenAI({ temperature: 0, openAIApiKey: process.env.OPENAI_API_KEY });
    const embeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
    return [
        new WebQA({ llm, embeddings }),
    ];
}

function initAgentWithTools(config: any) {
    const chat = new ChatOpenAI({
        modelName: config.settings.chat_model,
        temperature: parseFloat(config.settings.temperature),
        openAIApiKey: process.env.OPENAI_API_KEY,
    });
    const tools = loadTools(config);
    const agent = initializeAgentExecutorWithOptions(
        tools,
        chat,
        { agentType: "openai-functions", verbose: true },
    );
    return agent;
}

async function processMessagesFromFile(filePath: string, config: any) {
    const agent = await initAgentWithTools(config);
    const fileData = await readFile(filePath, 'utf-8');
    const messages = JSON.parse(fileData);
    for (const message of messages) {
        const responseMessage = await agent.run(message);
        console.log(responseMessage);
    }
}

async function main() {
    const argv = await yargs.option('config_file', { default: "config.json" }).option('message_file', { demandOption: true }).argv;
    const config = await loadConfig(argv.config_file);
    await processMessagesFromFile(argv.message_file as string, config);
}

main();
