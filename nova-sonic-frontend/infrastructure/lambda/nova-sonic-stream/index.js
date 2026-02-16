/**
 * Nova Sonic Stream Lambda Handler
 * 
 * Handles bidirectional streaming between AppSync Events and Bedrock Nova Sonic.
 * 
 * Architecture:
 *   Client (WebSocket via AppSync Events)
 *     ↕ Publish/Subscribe to channels
 *   This Lambda
 *     ↕ HTTP/2 Bidirectional Stream
 *   AWS Bedrock Nova Sonic (amazon.nova-2-sonic-v1:0)
 * 
 * Event Flow:
 *   1. Client subscribes to audio/{sessionId}/output
 *   2. Client publishes audio to audio/{sessionId}/input
 *   3. This Lambda receives audio, sends to Bedrock
 *   4. Bedrock returns audio/transcription
 *   5. Lambda publishes to audio/{sessionId}/output
 */

const { BedrockRuntimeClient, InvokeModelWithBidirectionalStreamCommand } = require('@aws-sdk/client-bedrock-runtime');
const { NodeHttp2Handler } = require('@smithy/node-http-handler');
const { DynamoDBClient, PutItemCommand, QueryCommand, UpdateItemCommand } = require('@aws-sdk/client-dynamodb');
const { marshall, unmarshall } = require('@aws-sdk/util-dynamodb');
const { randomUUID } = require('crypto');

// Configuration from environment
const BEDROCK_REGION = process.env.BEDROCK_REGION || 'us-east-1';
const NOVA_SONIC_MODEL_ID = process.env.NOVA_SONIC_MODEL_ID || 'amazon.nova-2-sonic-v1:0';
const CHAT_HISTORY_TABLE = process.env.CHAT_HISTORY_TABLE;
const SESSION_TABLE = process.env.SESSION_TABLE;
const APPSYNC_ENDPOINT = process.env.APPSYNC_ENDPOINT;
const APPSYNC_API_KEY = process.env.APPSYNC_API_KEY;

// Audio configuration
const MIN_AUDIO_CHUNKS_PER_BATCH = 10;
const MAX_AUDIO_CHUNKS_PER_BATCH = 20;
const AUDIO_BATCH_TIMEOUT_MS = 100;
const MAX_AUDIO_INPUT_QUEUE_SIZE = 200;
const BEDROCK_KEEP_ALIVE_INTERVAL_MS = 30000;

// Clients
const dynamoClient = new DynamoDBClient({ region: process.env.AWS_REGION });

// Active sessions Map (for Lambda warm instances)
const activeSessions = new Map();

/**
 * NovaStream class - Manages Bedrock Nova Sonic bidirectional stream
 */
class NovaStream {
  constructor(sessionId, voiceId, systemPrompt, tools, publishEvent) {
    this.sessionId = sessionId;
    this.voiceId = voiceId || 'tiffany';
    this.systemPrompt = systemPrompt;
    this.tools = tools || [];
    this.publishEvent = publishEvent;

    // Queues
    this.eventQueue = [];
    this.audioInputQueue = [];

    // IDs
    this.promptName = randomUUID();
    this.audioContentId = randomUUID();

    // State
    this.isActive = true;
    this.isAudioStarted = false;
    this.isProcessingAudio = false;
    this._stream = undefined;
    this._closedIntentionally = false;
    this._bedrockKeepAliveInterval = null;

    // Stats
    this.audioStats = {
      received: 0,
      queued: 0,
      sent: 0,
      dropped: 0,
      lastLogTime: Date.now(),
    };

    this.streamHealth = {
      openedAt: null,
      lastEventSentAt: null,
      lastEventReceivedAt: null,
    };

    // Create Bedrock client with proper timeout settings
    this.client = new BedrockRuntimeClient({
      region: BEDROCK_REGION,
      requestHandler: new NodeHttp2Handler({
        requestTimeout: 900000,  // 15 min
        sessionTimeout: 900000,  // 15 min
        disableConcurrentStreams: false,
        maxConcurrentStreams: 1,
      }),
    });

    this.modelId = NOVA_SONIC_MODEL_ID;
    console.log(`[NovaStream] Session ${sessionId} created with voice: ${voiceId}`);
  }

  get iterator() {
    if (!this._stream?.body) {
      throw new Error('Nova stream is not open yet!');
    }
    return this._stream.body;
  }

  get isProcessing() {
    return this.isActive && this.isAudioStarted;
  }

  get closedIntentionally() {
    return this._closedIntentionally;
  }

  _startBedrockKeepAlive() {
    this._bedrockKeepAliveInterval = setInterval(() => {
      if (!this.isActive || !this.isAudioStarted) return;

      if (this.eventQueue.length === 0) {
        const silentChunk = Buffer.alloc(320).toString('base64');
        this.eventQueue.push({
          event: {
            audioInput: {
              promptName: this.promptName,
              contentName: this.audioContentId,
              content: silentChunk,
            },
          },
        });
        console.log('[NovaStream] Keep-alive sent to Bedrock');
      }
    }, BEDROCK_KEEP_ALIVE_INTERVAL_MS);
  }

  _stopBedrockKeepAlive() {
    if (this._bedrockKeepAliveInterval) {
      clearInterval(this._bedrockKeepAliveInterval);
      this._bedrockKeepAliveInterval = null;
    }
  }

  async *getAsyncEventStream() {
    this.streamHealth.openedAt = Date.now();

    // Configuration events
    yield { event: { sessionStart: { inferenceConfiguration: this.getInferenceConfig() } } };
    yield { event: { promptStart: { promptName: this.promptName, textOutputConfiguration: { mediaType: 'text/plain' } } } };
    yield { event: { contentStart: { promptName: this.promptName, contentName: randomUUID(), type: 'TEXT', role: 'SYSTEM', textInputConfiguration: { mediaType: 'text/plain' } } } };
    yield { event: { textInput: { promptName: this.promptName, contentName: this.audioContentId, content: this.systemPrompt } } };
    yield { event: { contentEnd: { promptName: this.promptName, contentName: this.audioContentId } } };

    if (this.tools.length > 0) {
      yield { event: { toolUseContent: { promptName: this.promptName, tools: this.tools } } };
    }

    yield { event: { contentStart: { promptName: this.promptName, contentName: this.audioContentId, type: 'AUDIO', role: 'USER', audioInputConfiguration: { mediaType: 'audio/lpcm', sampleRateHertz: 16000, sampleSizeBits: 16, channelCount: 1, audioType: 'SPEECH', encoding: 'base64' } } } };

    this.isAudioStarted = true;
    this._startBedrockKeepAlive();
    console.log('[NovaStream] Audio streaming started');

    // Process event queue
    while (this.isActive) {
      if (this.eventQueue.length > 0) {
        const event = this.eventQueue.shift();
        this.audioStats.sent++;
        this.streamHealth.lastEventSentAt = Date.now();
        yield event;
      } else {
        await new Promise(resolve => setTimeout(resolve, 25));
      }
    }

    // Cleanup events
    yield { event: { contentEnd: { promptName: this.promptName, contentName: this.audioContentId } } };
    yield { event: { promptEnd: { promptName: this.promptName } } };
    yield { event: { sessionEnd: {} } };
  }

  getInferenceConfig() {
    return {
      maxTokens: 1024,
      topP: 0.9,
      temperature: 0.7,
      audio: {
        outputFormat: { sampleRateHertz: 24000, sampleSizeBits: 16, channelCount: 1, voiceId: this.voiceId, encoding: 'base64', mediaType: 'audio/lpcm' },
        inputFormat: { sampleRateHertz: 16000, sampleSizeBits: 16, channelCount: 1, audioType: 'SPEECH', encoding: 'base64', mediaType: 'audio/lpcm' },
      },
    };
  }

  async openStream() {
    const command = new InvokeModelWithBidirectionalStreamCommand({
      modelId: this.modelId,
      body: this.getAsyncEventStream(),
    });
    this._stream = await this.client.send(command);
    return this;
  }

  enqueueAudioInput(audioChunkBase64) {
    if (!this.isActive) return;

    this.audioStats.received++;

    if (this.audioInputQueue.length >= MAX_AUDIO_INPUT_QUEUE_SIZE) {
      this.audioStats.dropped++;
      this.audioInputQueue.shift();
    }
    this.audioInputQueue.push(audioChunkBase64);

    if (!this.isProcessingAudio) {
      this.processAudioQueue();
    }
  }

  processAudioQueue() {
    if (!this.isAudioStarted || this.audioInputQueue.length === 0) {
      this.isProcessingAudio = false;
      return;
    }

    this.isProcessingAudio = true;

    const chunksToSend = Math.min(this.audioInputQueue.length, MAX_AUDIO_CHUNKS_PER_BATCH);
    const batch = this.audioInputQueue.splice(0, chunksToSend);
    const combinedAudio = Buffer.concat(batch.map(chunk => Buffer.from(chunk, 'base64'))).toString('base64');

    this.eventQueue.push({
      event: {
        audioInput: {
          promptName: this.promptName,
          contentName: this.audioContentId,
          content: combinedAudio,
        },
      },
    });

    this.audioStats.queued += batch.length;

    if (this.audioInputQueue.length >= MIN_AUDIO_CHUNKS_PER_BATCH) {
      setImmediate(() => this.processAudioQueue());
    } else {
      setTimeout(() => this.processAudioQueue(), AUDIO_BATCH_TIMEOUT_MS);
    }
  }

  async close() {
    if (!this.isActive) return;

    this._closedIntentionally = true;
    this.isActive = false;
    this._stopBedrockKeepAlive();
    console.log(`[NovaStream] Session ${this.sessionId} closed`);
  }
}

/**
 * Process Nova Sonic response stream
 */
async function processNovaStream(novaStream, sessionId, publishEvent) {
  let currentRole = null;
  let currentContentId = null;
  let transcriptionBuffer = '';
  let responseBuffer = '';

  try {
    for await (const event of novaStream.iterator) {
      novaStream.streamHealth.lastEventReceivedAt = Date.now();

      if (event.chunk?.bytes) {
        const payload = JSON.parse(new TextDecoder().decode(event.chunk.bytes));

        // Content start - track role
        if (payload.event?.contentStart) {
          currentRole = payload.event.contentStart.role;
          currentContentId = payload.event.contentStart.contentName;
        }

        // Text output (transcription or response)
        if (payload.event?.textOutput) {
          const text = payload.event.textOutput.content || '';
          
          if (payload.event.textOutput.role === 'USER') {
            transcriptionBuffer += text;
          } else if (payload.event.textOutput.role === 'ASSISTANT') {
            responseBuffer += text;
          }
        }

        // Audio output - send to client
        if (payload.event?.audioOutput) {
          await publishEvent(sessionId, 'audio_output', {
            type: 'audioOutput',
            audio: payload.event.audioOutput.content,
          });
        }

        // Content end - emit complete transcription/response
        if (payload.event?.contentEnd) {
          if (currentRole === 'USER' && transcriptionBuffer) {
            await publishEvent(sessionId, 'transcription', {
              type: 'transcription',
              role: 'user',
              text: transcriptionBuffer.trim(),
            });
            
            // Save to DynamoDB
            await saveChatMessage(sessionId, 'USER', transcriptionBuffer.trim());
            transcriptionBuffer = '';
          } else if (currentRole === 'ASSISTANT' && responseBuffer) {
            await publishEvent(sessionId, 'response', {
              type: 'response',
              role: 'assistant',
              text: responseBuffer.trim(),
            });
            
            // Save to DynamoDB
            await saveChatMessage(sessionId, 'ASSISTANT', responseBuffer.trim());
            responseBuffer = '';
          }
          currentRole = null;
        }

        // Tool use
        if (payload.event?.toolUse) {
          await publishEvent(sessionId, 'tool_use', {
            type: 'toolUse',
            toolName: payload.event.toolUse.toolName,
            toolUseId: payload.event.toolUse.toolUseId,
            input: payload.event.toolUse.input,
          });
        }
      }
    }
  } catch (error) {
    console.error('[ProcessStream] Error:', error);
    await publishEvent(sessionId, 'error', {
      type: 'error',
      message: error.message,
      code: error.name,
    });
  }
}

/**
 * Save chat message to DynamoDB
 */
async function saveChatMessage(sessionId, role, content) {
  if (!CHAT_HISTORY_TABLE) return;

  try {
    const item = {
      sessionId,
      timestamp: Date.now(),
      role,
      content,
      ttl: Math.floor(Date.now() / 1000) + (7 * 24 * 60 * 60), // 7 days
    };

    await dynamoClient.send(new PutItemCommand({
      TableName: CHAT_HISTORY_TABLE,
      Item: marshall(item),
    }));
  } catch (error) {
    console.error('[DynamoDB] Error saving message:', error);
  }
}

/**
 * Load chat history from DynamoDB
 */
async function loadChatHistory(sessionId) {
  if (!CHAT_HISTORY_TABLE) return [];

  try {
    const result = await dynamoClient.send(new QueryCommand({
      TableName: CHAT_HISTORY_TABLE,
      KeyConditionExpression: 'sessionId = :sid',
      ExpressionAttributeValues: marshall({ ':sid': sessionId }),
      ScanIndexForward: true,
      Limit: 100,
    }));

    return (result.Items || []).map(item => unmarshall(item));
  } catch (error) {
    console.error('[DynamoDB] Error loading history:', error);
    return [];
  }
}

/**
 * Publish event to AppSync Events
 */
async function publishToAppSync(sessionId, eventType, data) {
  if (!APPSYNC_ENDPOINT || !APPSYNC_API_KEY) {
    console.log(`[AppSync] Would publish to audio/${sessionId}/output:`, eventType);
    return;
  }

  const channel = `/audio/${sessionId}/output`;
  const event = JSON.stringify({ ...data, timestamp: Date.now() });

  try {
    const response = await fetch(`${APPSYNC_ENDPOINT}/event`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': APPSYNC_API_KEY,
      },
      body: JSON.stringify({
        channel,
        events: [event],
      }),
    });

    if (!response.ok) {
      console.error('[AppSync] Publish failed:', response.status, await response.text());
    }
  } catch (error) {
    console.error('[AppSync] Publish error:', error);
  }
}

/**
 * Lambda Handler
 */
exports.handler = async (event, context) => {
  console.log('[Handler] Event:', JSON.stringify(event));

  const { action, sessionId, data } = typeof event.body === 'string' 
    ? JSON.parse(event.body) 
    : event;

  try {
    switch (action) {
      case 'start_session': {
        const { voiceId = 'tiffany', systemPrompt, tools = [] } = data || {};
        
        // Create publish function
        const publishEvent = (sid, type, payload) => publishToAppSync(sid, type, payload);
        
        // Create Nova stream
        const novaStream = new NovaStream(sessionId, voiceId, systemPrompt || getDefaultPrompt(), tools, publishEvent);
        
        // Open stream
        await novaStream.openStream();
        
        // Store session
        activeSessions.set(sessionId, novaStream);
        
        // Start processing in background
        processNovaStream(novaStream, sessionId, publishEvent).catch(err => {
          console.error('[Handler] Stream processing error:', err);
        });
        
        // Notify client
        await publishEvent(sessionId, 'session_started', {
          type: 'session_started',
          sessionId,
        });
        
        return {
          statusCode: 200,
          body: JSON.stringify({ success: true, sessionId }),
        };
      }

      case 'audio_input': {
        const novaStream = activeSessions.get(sessionId);
        if (!novaStream) {
          return {
            statusCode: 404,
            body: JSON.stringify({ error: 'Session not found' }),
          };
        }
        
        const { audio } = data;
        if (audio) {
          novaStream.enqueueAudioInput(audio);
        }
        
        return { statusCode: 200, body: JSON.stringify({ success: true }) };
      }

      case 'end_session': {
        const novaStream = activeSessions.get(sessionId);
        if (novaStream) {
          await novaStream.close();
          activeSessions.delete(sessionId);
        }
        
        return {
          statusCode: 200,
          body: JSON.stringify({ success: true, sessionId }),
        };
      }

      case 'get_history': {
        const history = await loadChatHistory(sessionId);
        return {
          statusCode: 200,
          body: JSON.stringify({ success: true, history }),
        };
      }

      default:
        return {
          statusCode: 400,
          body: JSON.stringify({ error: `Unknown action: ${action}` }),
        };
    }
  } catch (error) {
    console.error('[Handler] Error:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};

function getDefaultPrompt() {
  return `You are a warm, professional, and helpful AI assistant. Give accurate answers that sound natural, direct, and human.
Start by answering the user's question clearly in 1–2 sentences. Then, expand only enough to make the answer understandable, staying within 3–5 short sentences total.
Avoid sounding like a lecture or essay. Be conversational and friendly.`;
}
