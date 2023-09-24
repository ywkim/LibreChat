const GoogleSearchAPI = require('./GoogleSearch');
const HttpRequestTool = require('./HttpRequestTool');
const AIPluginTool = require('./AIPluginTool');
const OpenAICreateImage = require('./DALL-E');
const StructuredSD = require('./structured/StableDiffusion');
const StableDiffusionAPI = require('./StableDiffusion');
const WolframAlphaAPI = require('./Wolfram');
const StructuredWolfram = require('./structured/Wolfram');
const SelfReflectionTool = require('./SelfReflection');
const WebQA = require('./WebQA');
const SearchQA = require('./SearchQA');
const BookQA = require('./BookQA');
const AzureCognitiveSearch = require('./AzureCognitiveSearch');
const StructuredACS = require('./structured/AzureCognitiveSearch');
const ChatTool = require('./structured/ChatTool');
const E2BTools = require('./structured/E2BTools');
const CodeSherpa = require('./structured/CodeSherpa');
const CodeSherpaTools = require('./structured/CodeSherpaTools');
const availableTools = require('./manifest.json');
const CodeInterpreter = require('./CodeInterpreter');
const CodeBrew = require('./CodeBrew');

module.exports = {
  availableTools,
  GoogleSearchAPI,
  HttpRequestTool,
  AIPluginTool,
  OpenAICreateImage,
  StableDiffusionAPI,
  StructuredSD,
  WolframAlphaAPI,
  StructuredWolfram,
  SelfReflectionTool,
  WebQA,
  SearchQA,
  BookQA,
  AzureCognitiveSearch,
  StructuredACS,
  E2BTools,
  ChatTool,
  CodeSherpa,
  CodeSherpaTools,
  CodeInterpreter,
  CodeBrew,
};
