const { Tiktoken } = require('tiktoken/lite');
const { load } = require('tiktoken/load');
const registry = require('tiktoken/registry.json');
const models = require('tiktoken/model_to_encoding.json');

const countTokens = async (text = '', modelName = 'gpt-3.5-turbo') => {
  let encoder = null;
  try {
    const model = await load(registry[models[modelName]]);
    encoder = new Tiktoken(model.bpe_ranks, model.special_tokens, model.pat_str);
    const tokens = encoder.encode(text);
    encoder.free();
    return tokens.length;
  } catch (e) {
    console.error(e);
    if (encoder) {
      encoder.free();
    }
    return 0;
  }
};

module.exports = countTokens;
