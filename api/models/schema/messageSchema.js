const mongoose = require('mongoose');
const mongoMeili = require('../plugins/mongoMeili');
const messageSchema = mongoose.Schema(
  {
    messageId: {
      type: String,
      unique: true,
      required: true,
      index: true,
      meiliIndex: true,
    },
    conversationId: {
      type: String,
      required: true,
      meiliIndex: true,
    },
    user: {
      type: String,
      index: true,
      default: null,
    },
    model: {
      type: String,
    },
    conversationSignature: {
      type: String,
      // required: true
    },
    clientId: {
      type: String,
    },
    invocationId: {
      type: Number,
    },
    parentMessageId: {
      type: String,
      // required: true
    },
    tokenCount: {
      type: Number,
    },
    summaryTokenCount: {
      type: Number,
    },
    sender: {
      type: String,
      required: true,
      meiliIndex: true,
    },
    text: {
      type: String,
      required: true,
      meiliIndex: true,
    },
    summary: {
      type: String,
    },
    isCreatedByUser: {
      type: Boolean,
      required: true,
      default: false,
    },
    isEdited: {
      type: Boolean,
      default: false,
    },
    unfinished: {
      type: Boolean,
      default: false,
    },
    cancelled: {
      type: Boolean,
      default: false,
    },
    error: {
      type: Boolean,
      default: false,
    },
    finish_reason: {
      type: String,
    },
    _meiliIndex: {
      type: Boolean,
      required: false,
      select: false,
      default: false,
    },
    plugin: {
      latest: {
        type: String,
        required: false,
      },
      inputs: {
        type: [mongoose.Schema.Types.Mixed],
        required: false,
      },
      outputs: {
        type: String,
        required: false,
      },
    },
    plugins: [{ type: mongoose.Schema.Types.Mixed }],
  },
  { timestamps: true },
);

if (process.env.MEILI_HOST && process.env.MEILI_MASTER_KEY) {
  messageSchema.plugin(mongoMeili, {
    host: process.env.MEILI_HOST,
    apiKey: process.env.MEILI_MASTER_KEY,
    indexName: 'messages',
    primaryKey: 'messageId',
  });
}

messageSchema.index({ createdAt: 1 });

const Message = mongoose.models.Message || mongoose.model('Message', messageSchema);

module.exports = Message;
