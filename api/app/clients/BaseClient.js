const crypto = require('crypto');
const TextStream = require('./TextStream');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { ChatOpenAI } = require('langchain/chat_models/openai');
const { loadSummarizationChain } = require('langchain/chains');
const { getConvo, getMessages, saveMessage, updateMessage, saveConvo } = require('../../models');
const { addSpaceIfNeeded } = require('../../server/utils');
const { refinePrompt } = require('./prompts');

class BaseClient {
  constructor(apiKey, options = {}) {
    this.apiKey = apiKey;
    this.sender = options.sender ?? 'AI';
    this.contextStrategy = null;
    this.currentDateString = new Date().toLocaleDateString('en-us', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  }

  setOptions() {
    throw new Error('Method \'setOptions\' must be implemented.');
  }

  getCompletion() {
    throw new Error('Method \'getCompletion\' must be implemented.');
  }

  async sendCompletion() {
    throw new Error('Method \'sendCompletion\' must be implemented.');
  }

  getSaveOptions() {
    throw new Error('Subclasses must implement getSaveOptions');
  }

  async buildMessages() {
    throw new Error('Subclasses must implement buildMessages');
  }

  getBuildMessagesOptions() {
    throw new Error('Subclasses must implement getBuildMessagesOptions');
  }

  async generateTextStream(text, onProgress, options = {}) {
    const stream = new TextStream(text, options);
    await stream.processTextStream(onProgress);
  }

  async setMessageOptions(opts = {}) {
    if (opts && typeof opts === 'object') {
      this.setOptions(opts);
    }

    const { isEdited, isContinued } = opts;
    const user = opts.user ?? null;
    this.user = user;
    const saveOptions = this.getSaveOptions();
    this.abortController = opts.abortController ?? new AbortController();
    const conversationId = opts.conversationId ?? crypto.randomUUID();
    const parentMessageId = opts.parentMessageId ?? '00000000-0000-0000-0000-000000000000';
    const userMessageId = opts.overrideParentMessageId ?? crypto.randomUUID();
    let responseMessageId = opts.responseMessageId ?? crypto.randomUUID();
    let head = isEdited ? responseMessageId : parentMessageId;
    this.currentMessages = (await this.loadHistory(conversationId, head)) ?? [];

    if (isEdited && !isContinued) {
      responseMessageId = crypto.randomUUID();
      head = responseMessageId;
      this.currentMessages[this.currentMessages.length - 1].messageId = head;
    }

    return {
      ...opts,
      user,
      head,
      conversationId,
      parentMessageId,
      userMessageId,
      responseMessageId,
      saveOptions,
    };
  }

  createUserMessage({ messageId, parentMessageId, conversationId, text }) {
    return {
      messageId,
      parentMessageId,
      conversationId,
      sender: 'User',
      text,
      isCreatedByUser: true,
    };
  }

  async handleStartMethods(message, opts) {
    const {
      user,
      head,
      conversationId,
      parentMessageId,
      userMessageId,
      responseMessageId,
      saveOptions,
    } = await this.setMessageOptions(opts);

    const userMessage = opts.isEdited
      ? this.currentMessages[this.currentMessages.length - 2]
      : this.createUserMessage({
        messageId: userMessageId,
        parentMessageId,
        conversationId,
        text: message,
      });

    if (typeof opts?.getIds === 'function') {
      opts.getIds({
        userMessage,
        conversationId,
        responseMessageId,
      });
    }

    if (typeof opts?.onStart === 'function') {
      opts.onStart(userMessage);
    }

    return {
      ...opts,
      user,
      head,
      conversationId,
      responseMessageId,
      saveOptions,
      userMessage,
    };
  }

  addInstructions(messages, instructions) {
    const payload = [];
    if (!instructions) {
      return messages;
    }
    if (messages.length > 1) {
      payload.push(...messages.slice(0, -1));
    }

    payload.push(instructions);

    if (messages.length > 0) {
      payload.push(messages[messages.length - 1]);
    }

    return payload;
  }

  async handleTokenCountMap(tokenCountMap) {
    if (this.currentMessages.length === 0) {
      return;
    }

    for (let i = 0; i < this.currentMessages.length; i++) {
      // Skip the last message, which is the user message.
      if (i === this.currentMessages.length - 1) {
        break;
      }

      const message = this.currentMessages[i];
      const { messageId } = message;
      const update = {};

      if (messageId === tokenCountMap.refined?.messageId) {
        if (this.options.debug) {
          console.debug(`Adding refined props to ${messageId}.`);
        }

        update.refinedMessageText = tokenCountMap.refined.content;
        update.refinedTokenCount = tokenCountMap.refined.tokenCount;
      }

      if (message.tokenCount && !update.refinedTokenCount) {
        if (this.options.debug) {
          console.debug(`Skipping ${messageId}: already had a token count.`);
        }
        continue;
      }

      const tokenCount = tokenCountMap[messageId];
      if (tokenCount) {
        message.tokenCount = tokenCount;
        update.tokenCount = tokenCount;
        await this.updateMessageInDatabase({ messageId, ...update });
      }
    }
  }

  concatenateMessages(messages) {
    return messages.reduce((acc, message) => {
      const nameOrRole = message.name ?? message.role;
      return acc + `${nameOrRole}:\n${message.content}\n\n`;
    }, '');
  }

  async refineMessages(messagesToRefine, remainingContextTokens) {
    const model = new ChatOpenAI({ temperature: 0 });
    const chain = loadSummarizationChain(model, {
      type: 'refine',
      verbose: this.options.debug,
      refinePrompt,
    });
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1500,
      chunkOverlap: 100,
    });
    const userMessages = this.concatenateMessages(
      messagesToRefine.filter((m) => m.role === 'user'),
    );
    const assistantMessages = this.concatenateMessages(
      messagesToRefine.filter((m) => m.role !== 'user'),
    );
    const userDocs = await splitter.createDocuments([userMessages], [], {
      chunkHeader: 'DOCUMENT NAME: User Message\n\n---\n\n',
      appendChunkOverlapHeader: true,
    });
    const assistantDocs = await splitter.createDocuments([assistantMessages], [], {
      chunkHeader: 'DOCUMENT NAME: Assistant Message\n\n---\n\n',
      appendChunkOverlapHeader: true,
    });
    // const chunkSize = Math.round(concatenatedMessages.length / 512);
    const input_documents = userDocs.concat(assistantDocs);
    if (this.options.debug) {
      console.debug('Refining messages...');
    }
    try {
      const res = await chain.call({
        input_documents,
        signal: this.abortController.signal,
      });

      const refinedMessage = {
        role: 'assistant',
        content: res.output_text,
        tokenCount: this.getTokenCount(res.output_text),
      };

      if (this.options.debug) {
        console.debug('Refined messages', refinedMessage);
        console.debug(
          `remainingContextTokens: ${remainingContextTokens}, after refining: ${
            remainingContextTokens - refinedMessage.tokenCount
          }`,
        );
      }

      return refinedMessage;
    } catch (e) {
      console.error('Error refining messages');
      console.error(e);
      return null;
    }
  }

  /**
   * This method processes an array of messages and returns a context of messages that fit within a token limit.
   * It iterates over the messages from newest to oldest, adding them to the context until the token limit is reached.
   * If the token limit would be exceeded by adding a message, that message and possibly the previous one are added to a separate array of messages to refine.
   * The method uses `push` and `pop` operations for efficient array manipulation, and reverses the arrays at the end to maintain the original order of the messages.
   * The method also includes a mechanism to avoid blocking the event loop by waiting for the next tick after each iteration.
   *
   * @param {Array} messages - An array of messages, each with a `tokenCount` property. The messages should be ordered from oldest to newest.
   * @returns {Object} An object with three properties: `context`, `remainingContextTokens`, and `messagesToRefine`. `context` is an array of messages that fit within the token limit. `remainingContextTokens` is the number of tokens remaining within the limit after adding the messages to the context. `messagesToRefine` is an array of messages that were not added to the context because they would have exceeded the token limit.
   */
  async getMessagesWithinTokenLimit(messages) {
    // Every reply is primed with <|start|>assistant<|message|>, so we
    // start with 3 tokens for the label after all messages have been counted.
    let currentTokenCount = 3;
    let context = [];
    let messagesToRefine = [];
    let refineIndex = -1;
    let remainingContextTokens = this.maxContextTokens;

    for (let i = messages.length - 1; i >= 0; i--) {
      const message = messages[i];
      const newTokenCount = currentTokenCount + message.tokenCount;
      const exceededLimit = newTokenCount > this.maxContextTokens;
      let shouldRefine = exceededLimit && this.shouldRefineContext;
      let refineNextMessage = i !== 0 && i !== 1 && context.length > 0;

      if (shouldRefine) {
        messagesToRefine.push(message);

        if (refineIndex === -1) {
          refineIndex = i;
        }

        if (refineNextMessage) {
          refineIndex = i + 1;
          const removedMessage = context.pop();
          messagesToRefine.push(removedMessage);
          currentTokenCount -= removedMessage.tokenCount;
          remainingContextTokens = this.maxContextTokens - currentTokenCount;
          refineNextMessage = false;
        }

        continue;
      } else if (exceededLimit) {
        break;
      }

      context.push(message);
      currentTokenCount = newTokenCount;
      remainingContextTokens = this.maxContextTokens - currentTokenCount;
      await new Promise((resolve) => setImmediate(resolve));
    }

    return {
      context: context.reverse(),
      remainingContextTokens,
      messagesToRefine: messagesToRefine.reverse(),
      refineIndex,
    };
  }

  async handleContextStrategy({ instructions, orderedMessages, formattedMessages }) {
    let payload = this.addInstructions(formattedMessages, instructions);
    let orderedWithInstructions = this.addInstructions(orderedMessages, instructions);
    let { context, remainingContextTokens, messagesToRefine, refineIndex } =
      await this.getMessagesWithinTokenLimit(payload);

    payload = context;
    let refinedMessage;

    // if (messagesToRefine.length > 0) {
    //   refinedMessage = await this.refineMessages(messagesToRefine, remainingContextTokens);
    //   payload.unshift(refinedMessage);
    //   remainingContextTokens -= refinedMessage.tokenCount;
    // }
    // if (remainingContextTokens <= instructions?.tokenCount) {
    //   if (this.options.debug) {
    //     console.debug(`Remaining context (${remainingContextTokens}) is less than instructions token count: ${instructions.tokenCount}`);
    //   }

    //   ({ context, remainingContextTokens, messagesToRefine, refineIndex } = await this.getMessagesWithinTokenLimit(payload));
    //   payload = context;
    // }

    // Calculate the difference in length to determine how many messages were discarded if any
    let diff = orderedWithInstructions.length - payload.length;

    if (this.options.debug) {
      console.debug('<---------------------------------DIFF--------------------------------->');
      console.debug(
        `Difference between payload (${payload.length}) and orderedWithInstructions (${orderedWithInstructions.length}): ${diff}`,
      );
      console.debug(
        'remainingContextTokens, this.maxContextTokens (1/2)',
        remainingContextTokens,
        this.maxContextTokens,
      );
    }

    // If the difference is positive, slice the orderedWithInstructions array
    if (diff > 0) {
      orderedWithInstructions = orderedWithInstructions.slice(diff);
    }

    if (messagesToRefine.length > 0) {
      refinedMessage = await this.refineMessages(messagesToRefine, remainingContextTokens);
      payload.unshift(refinedMessage);
      remainingContextTokens -= refinedMessage.tokenCount;
    }

    if (this.options.debug) {
      console.debug(
        'remainingContextTokens, this.maxContextTokens (2/2)',
        remainingContextTokens,
        this.maxContextTokens,
      );
    }

    let tokenCountMap = orderedWithInstructions.reduce((map, message, index) => {
      if (!message.messageId) {
        return map;
      }

      if (index === refineIndex) {
        map.refined = { ...refinedMessage, messageId: message.messageId };
      }

      map[message.messageId] = payload[index].tokenCount;
      return map;
    }, {});

    const promptTokens = this.maxContextTokens - remainingContextTokens;

    if (this.options.debug) {
      console.debug('<-------------------------PAYLOAD/TOKEN COUNT MAP------------------------->');
      // console.debug('Payload:', payload);
      console.debug('Token Count Map:', tokenCountMap);
      console.debug('Prompt Tokens', promptTokens, remainingContextTokens, this.maxContextTokens);
    }

    return { payload, tokenCountMap, promptTokens, messages: orderedWithInstructions };
  }

  async sendMessage(message, opts = {}) {
    const { user, head, isEdited, conversationId, responseMessageId, saveOptions, userMessage } =
      await this.handleStartMethods(message, opts);

    const { generation = '' } = opts;

    // It's not necessary to push to currentMessages
    // depending on subclass implementation of handling messages
    // When this is an edit, all messages are already in currentMessages, both user and response
    if (isEdited) {
      let latestMessage = this.currentMessages[this.currentMessages.length - 1];
      if (!latestMessage) {
        latestMessage = {
          messageId: responseMessageId,
          conversationId,
          parentMessageId: userMessage.messageId,
          isCreatedByUser: false,
          model: this.modelOptions.model,
          sender: this.sender,
          text: generation,
        };
        this.currentMessages.push(userMessage, latestMessage);
      } else {
        latestMessage.text = generation;
      }
    } else {
      this.currentMessages.push(userMessage);
    }

    let {
      prompt: payload,
      tokenCountMap,
      promptTokens,
    } = await this.buildMessages(
      this.currentMessages,
      // When the userMessage is pushed to currentMessages, the parentMessage is the userMessageId.
      // this only matters when buildMessages is utilizing the parentMessageId, and may vary on implementation
      isEdited ? head : userMessage.messageId,
      this.getBuildMessagesOptions(opts),
    );

    if (this.options.debug) {
      console.debug('payload');
      console.debug(payload);
    }

    if (tokenCountMap) {
      console.dir(tokenCountMap, { depth: null });
      if (tokenCountMap[userMessage.messageId]) {
        userMessage.tokenCount = tokenCountMap[userMessage.messageId];
        console.log('userMessage.tokenCount', userMessage.tokenCount);
        console.log('userMessage', userMessage);
      }

      payload = payload.map((message) => {
        const messageWithoutTokenCount = message;
        delete messageWithoutTokenCount.tokenCount;
        return messageWithoutTokenCount;
      });
      this.handleTokenCountMap(tokenCountMap);
    }

    if (!isEdited) {
      await this.saveMessageToDatabase(userMessage, saveOptions, user);
    }

    const responseMessage = {
      messageId: responseMessageId,
      conversationId,
      parentMessageId: userMessage.messageId,
      isCreatedByUser: false,
      isEdited,
      model: this.modelOptions.model,
      sender: this.sender,
      text: addSpaceIfNeeded(generation) + (await this.sendCompletion(payload, opts)),
      promptTokens,
    };

    if (tokenCountMap && this.getTokenCountForResponse) {
      responseMessage.tokenCount = this.getTokenCountForResponse(responseMessage);
      responseMessage.completionTokens = responseMessage.tokenCount;
    }
    await this.saveMessageToDatabase(responseMessage, saveOptions, user);
    delete responseMessage.tokenCount;
    return responseMessage;
  }

  async getConversation(conversationId, user = null) {
    return await getConvo(user, conversationId);
  }

  async loadHistory(conversationId, parentMessageId = null) {
    if (this.options.debug) {
      console.debug('Loading history for conversation', conversationId, parentMessageId);
    }

    const messages = (await getMessages({ conversationId })) ?? [];

    if (messages.length === 0) {
      return [];
    }

    let mapMethod = null;
    if (this.getMessageMapMethod) {
      mapMethod = this.getMessageMapMethod();
    }

    return this.constructor.getMessagesForConversation(messages, parentMessageId, mapMethod);
  }

  async saveMessageToDatabase(message, endpointOptions, user = null) {
    await saveMessage({ ...message, user, unfinished: false, cancelled: false });
    await saveConvo(user, {
      conversationId: message.conversationId,
      endpoint: this.options.endpoint,
      ...endpointOptions,
    });
  }

  async updateMessageInDatabase(message) {
    await updateMessage(message);
  }

  /**
   * Iterate through messages, building an array based on the parentMessageId.
   * Each message has an id and a parentMessageId. The parentMessageId is the id of the message that this message is a reply to.
   * @param messages
   * @param parentMessageId
   * @returns {*[]} An array containing the messages in the order they should be displayed, starting with the root message.
   */
  static getMessagesForConversation(messages, parentMessageId, mapMethod = null) {
    if (!messages || messages.length === 0) {
      return [];
    }

    const orderedMessages = [];
    let currentMessageId = parentMessageId;
    while (currentMessageId) {
      const message = messages.find((msg) => {
        const messageId = msg.messageId ?? msg.id;
        return messageId === currentMessageId;
      });
      if (!message) {
        break;
      }
      orderedMessages.unshift(message);
      currentMessageId = message.parentMessageId;
    }

    if (mapMethod) {
      return orderedMessages.map(mapMethod);
    }

    return orderedMessages;
  }

  /**
   * Algorithm adapted from "6. Counting tokens for chat API calls" of
   * https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
   *
   * An additional 3 tokens need to be added for assistant label priming after all messages have been counted.
   *
   * @param {Object} message
   */
  getTokenCountForMessage(message) {
    // Note: gpt-3.5-turbo and gpt-4 may update over time. Use default for these as well as for unknown models
    let tokensPerMessage = 3;
    let tokensPerName = 1;

    if (this.modelOptions.model === 'gpt-3.5-turbo-0301') {
      tokensPerMessage = 4;
      tokensPerName = -1;
    }

    let numTokens = tokensPerMessage;
    for (let [key, value] of Object.entries(message)) {
      numTokens += this.getTokenCount(value);
      if (key === 'name') {
        numTokens += tokensPerName;
      }
    }

    return numTokens;
  }

  async sendPayload(payload, opts = {}) {
    if (opts && typeof opts === 'object') {
      this.setOptions(opts);
    }

    return await this.sendCompletion(payload, opts);
  }
}

module.exports = BaseClient;
