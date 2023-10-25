// From https://platform.openai.com/docs/api-reference/images/create
// To use this tool, you must pass in a configured OpenAIApi object.
const fs = require('fs');
const OpenAI = require('openai');
// const { genAzureEndpoint } = require('../../../utils/genAzureEndpoints');
const { Tool } = require('langchain/tools');
const saveImageFromUrl = require('./saveImageFromUrl');
const path = require('path');

class OpenAICreateImage extends Tool {
  constructor(fields = {}) {
    super();

    let apiKey = fields.DALLE_API_KEY || this.getApiKey();
    // let azureKey = fields.AZURE_API_KEY || process.env.AZURE_API_KEY;
    let config = { apiKey };

    // if (azureKey) {
    //   apiKey = azureKey;
    //   const azureConfig = {
    //     apiKey,
    //     azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME || fields.azureOpenAIApiInstanceName,
    //     azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME || fields.azureOpenAIApiDeploymentName,
    //     azureOpenAIApiVersion: process.env.AZURE_OPENAI_API_VERSION || fields.azureOpenAIApiVersion
    //   };
    //   config = {
    //     apiKey,
    //     basePath: genAzureEndpoint({
    //       ...azureConfig,
    //     }),
    //     baseOptions: {
    //       headers: { 'api-key': apiKey },
    //       params: {
    //         'api-version': azureConfig.azureOpenAIApiVersion // this might change. I got the current value from the sample code at https://oai.azure.com/portal/chat
    //       }
    //     }
    //   };
    // }
    this.openai = new OpenAI(config);
    this.name = 'dall-e';
    this.description = `You can generate images with 'dall-e'. This tool is exclusively for visual content.
Guidelines:
- Visually describe the moods, details, structures, styles, and/or proportions of the image. Remember, the focus is on visual attributes.
- Craft your input by "showing" and not "telling" the imagery. Think in terms of what you'd want to see in a photograph or a painting.
- It's best to follow this format for image creation. Come up with the optional inputs yourself if none are given:
"Subject: [subject], Style: [style], Color: [color], Details: [details], Emotion: [emotion]"
- Generate images only once per human query unless explicitly requested by the user`;
  }

  getApiKey() {
    const apiKey = process.env.DALLE_API_KEY || '';
    if (!apiKey) {
      throw new Error('Missing DALLE_API_KEY environment variable.');
    }
    return apiKey;
  }

  replaceUnwantedChars(inputString) {
    return inputString
      .replace(/\r\n|\r|\n/g, ' ')
      .replace('"', '')
      .trim();
  }

  getMarkdownImageUrl(imageName) {
    const imageUrl = path
      .join(this.relativeImageUrl, imageName)
      .replace(/\\/g, '/')
      .replace('public/', '');
    return `![generated image](/${imageUrl})`;
  }

  async _call(input) {
    const resp = await this.openai.images.generate({
      prompt: this.replaceUnwantedChars(input),
      // TODO: Future idea -- could we ask an LLM to extract these arguments from an input that might contain them?
      n: 1,
      // size: '1024x1024'
      size: '512x512',
    });

    const theImageUrl = resp.data[0].url;

    if (!theImageUrl) {
      throw new Error('No image URL returned from OpenAI API.');
    }

    const regex = /img-[\w\d]+.png/;
    const match = theImageUrl.match(regex);
    let imageName = '1.png';

    if (match) {
      imageName = match[0];
      console.log(imageName); // Output: img-lgCf7ppcbhqQrz6a5ear6FOb.png
    } else {
      console.log('No image name found in the string.');
    }

    this.outputPath = path.resolve(__dirname, '..', '..', '..', '..', 'client', 'public', 'images');
    const appRoot = path.resolve(__dirname, '..', '..', '..', '..', 'client');
    this.relativeImageUrl = path.relative(appRoot, this.outputPath);

    // Check if directory exists, if not create it
    if (!fs.existsSync(this.outputPath)) {
      fs.mkdirSync(this.outputPath, { recursive: true });
    }

    try {
      await saveImageFromUrl(theImageUrl, this.outputPath, imageName);
      this.result = this.getMarkdownImageUrl(imageName);
    } catch (error) {
      console.error('Error while saving the image:', error);
      this.result = theImageUrl;
    }

    return this.result;
  }
}

module.exports = OpenAICreateImage;
