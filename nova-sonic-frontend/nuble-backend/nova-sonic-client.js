/**
 * Nova Sonic Bidirectional Streaming Client
 * Based on aws-samples/sample-voicebot-nova-sonic
 * 
 * Manages bidirectional streaming sessions with Amazon Nova Sonic via AWS Bedrock.
 * Uses HTTP/2 for bidirectional streaming with async iterator pattern for event queue.
 */

const {
  BedrockRuntimeClient,
  InvokeModelWithBidirectionalStreamCommand,
} = require('@aws-sdk/client-bedrock-runtime');
const { NodeHttp2Handler } = require('@smithy/node-http-handler');
const { randomUUID } = require('crypto');
const { Buffer } = require('buffer');

// ==================== Constants ====================

const NOVA_SONIC_MODEL_ID = 'amazon.nova-2-sonic-v1:0';

const DefaultAudioInputConfiguration = {
  mediaType: 'audio/lpcm',
  sampleRateHertz: 16000,
  sampleSizeBits: 16,
  channelCount: 1,
  audioType: 'SPEECH',
  encoding: 'base64',
};

const DefaultAudioOutputConfiguration = {
  mediaType: 'audio/lpcm',
  sampleRateHertz: 24000,
  sampleSizeBits: 16,
  channelCount: 1,
  voiceId: 'matthew',
};

const DefaultTextConfiguration = {
  mediaType: 'text/plain',
};

// ==================== StreamSession ====================

/**
 * Represents a single streaming session. Wraps the client's methods
 * for a specific session ID with a fluent API.
 */
class StreamSession {
  constructor(sessionId, client) {
    this.sessionId = sessionId;
    this.client = client;
    this.voiceId = undefined;
    this.outputSampleRate = 24000;
    this.audioBufferQueue = [];
    this.audioQueueHead = 0;
    this.maxQueueSize = 200;
    this.isProcessingAudio = false;
    this.isActive = true;
  }

  /**
   * Register event handlers for this session
   * @param {string} eventType 
   * @param {Function} handler 
   * @returns {StreamSession} for chaining
   */
  onEvent(eventType, handler) {
    this.client.registerEventHandler(this.sessionId, eventType, handler);
    return this;
  }

  /**
   * Setup session start and prompt start events
   */
  async setupSessionAndPromptStart(voiceId, outputSampleRate = 24000) {
    this.voiceId = voiceId;
    this.outputSampleRate = outputSampleRate;
    this.client.setupSessionStartEvent(this.sessionId);
    this.client.setupPromptStartEvent(this.sessionId, voiceId, outputSampleRate);
  }

  /**
   * Setup the system prompt
   */
  async setupSystemPrompt(textConfig, systemPromptContent, voiceId) {
    if (voiceId) this.voiceId = voiceId;
    this.client.setupSystemPromptEvent(
      this.sessionId,
      textConfig || DefaultTextConfiguration,
      systemPromptContent
    );
  }

  /**
   * Setup audio start event
   */
  async setupStartAudio(audioConfig) {
    this.client.setupStartAudioEvent(this.sessionId, audioConfig || DefaultAudioInputConfiguration);
  }

  /**
   * Stream audio data to the model
   */
  async streamAudio(audioData) {
    if (!this.isActive) return;
    
    // Queue audio chunks with index-based dequeue (avoids O(n) shift)
    const queueLen = this.audioBufferQueue.length - this.audioQueueHead;
    if (queueLen >= this.maxQueueSize) {
      // Drop oldest chunk if queue is full
      this.audioQueueHead++;
    }
    this.audioBufferQueue.push(audioData);
    
    // Process queue
    if (!this.isProcessingAudio) {
      this.isProcessingAudio = true;
      try {
        while (this.audioQueueHead < this.audioBufferQueue.length && this.isActive) {
          const chunk = this.audioBufferQueue[this.audioQueueHead++];
          await this.client.streamAudioChunk(this.sessionId, chunk);
        }
        // Compact when fully drained
        if (this.audioQueueHead > 0) {
          this.audioBufferQueue = [];
          this.audioQueueHead = 0;
        }
      } finally {
        this.isProcessingAudio = false;
      }
    }
  }

  /**
   * Send content end event
   */
  async endAudioContent() {
    await this.client.sendContentEnd(this.sessionId);
  }

  /**
   * Send prompt end event
   */
  async endPrompt() {
    await this.client.sendPromptEnd(this.sessionId);
  }

  /**
   * Close the session
   */
  async close() {
    this.isActive = false;
    this.audioBufferQueue = [];
    this.audioQueueHead = 0;
    await this.client.sendSessionEnd(this.sessionId);
  }
}

// ==================== NovaSonicBidirectionalStreamClient ====================

/**
 * Main client that manages multiple concurrent Bedrock Nova Sonic sessions.
 * Uses an async iterator pattern with event queue for bidirectional streaming.
 */
class NovaSonicBidirectionalStreamClient {
  constructor(config = {}) {
    const nodeHttp2Handler = new NodeHttp2Handler({
      requestTimeout: 300000,
      sessionTimeout: 300000,
      disableConcurrentStreams: false,
      maxConcurrentStreams: 20,
      ...(config.requestHandlerConfig || {}),
    });

    this.bedrockRuntimeClient = new BedrockRuntimeClient({
      ...(config.clientConfig || {}),
      region: config.clientConfig?.region || 'us-east-1',
      requestHandler: nodeHttp2Handler,
    });

    this.inferenceConfig = config.inferenceConfig || {
      maxTokens: 1024,
      topP: 0.9,
      temperature: 0.7,
    };

    this.turnDetectionConfig = config.turnDetectionConfig;
    
    // Active sessions map
    this.activeSessions = new Map();
    this.sessionLastActivity = new Map();
    this.sessionCleanupInProgress = new Set();
  }

  // ---- Session Lifecycle ----

  isSessionActive(sessionId) {
    const session = this.activeSessions.get(sessionId);
    return session ? session.isActive : false;
  }

  getActiveSessions() {
    return Array.from(this.activeSessions.keys());
  }

  getLastActivityTime(sessionId) {
    return this.sessionLastActivity.get(sessionId) || 0;
  }

  updateSessionActivity(sessionId) {
    this.sessionLastActivity.set(sessionId, Date.now());
  }

  isCleanupInProgress(sessionId) {
    return this.sessionCleanupInProgress.has(sessionId);
  }

  /**
   * Create a new streaming session
   */
  createStreamSession(sessionId = randomUUID(), config = {}) {
    if (this.activeSessions.has(sessionId)) {
      throw new Error(`Stream session with ID ${sessionId} already exists`);
    }

    const session = {
      queue: [],
      queueHead: 0,
      queueSignal: null,    // Will be set in async iterator
      queueResolve: null,   // Promise resolver for queue wait
      closeRequested: false,
      responseHandlers: new Map(),
      promptName: randomUUID(),
      inferenceConfig: config.inferenceConfig || this.inferenceConfig,
      turnDetectionConfig: config.turnDetectionConfig || this.turnDetectionConfig,
      toolUseContent: null,
      toolUseId: '',
      toolName: '',
      isActive: true,
      isPromptStartSent: false,
      isAudioContentStartSent: false,
      audioContentId: randomUUID(),
    };

    this.activeSessions.set(sessionId, session);
    this.sessionLastActivity.set(sessionId, Date.now());

    return new StreamSession(sessionId, this);
  }

  // ---- Event Queue ----

  /**
   * Add an event to a session's queue
   */
  addEventToSessionQueue(sessionId, event) {
    const session = this.activeSessions.get(sessionId);
    if (!session || !session.isActive) return;

    this.updateSessionActivity(sessionId);
    session.queue.push(event);
    
    // Signal the async iterator that data is available
    if (session.queueResolve) {
      session.queueResolve();
      session.queueResolve = null;
    }
  }

  /**
   * Create an async iterable for a session's event queue.
   * This is fed into InvokeModelWithBidirectionalStreamCommand.
   * Events must be yielded as {chunk: {bytes: Uint8Array}} format.
   */
  createSessionAsyncIterable(sessionId) {
    const self = this;
    
    return {
      [Symbol.asyncIterator]() {
        return {
          async next() {
            const session = self.activeSessions.get(sessionId);
            if (!session.isActive || session.closeRequested) {
              return { value: undefined, done: true };
            }

            // Wait for items in the queue
            while (session.queueHead >= session.queue.length) {
              if (!session.isActive || session.closeRequested) {
                return { value: undefined, done: true };
              }
              
              // Wait for signal
              await new Promise((resolve) => {
                session.queueResolve = resolve;
                // Timeout to prevent infinite waits
                setTimeout(resolve, 5000);
              });

              if (!session.isActive || session.closeRequested) {
                return { value: undefined, done: true };
              }
            }

            const event = session.queue[session.queueHead++];
            
            // Compact when we've consumed many items
            if (session.queueHead > 100) {
              session.queue = session.queue.slice(session.queueHead);
              session.queueHead = 0;
            }
            
            // Encode the event as JSON bytes in the expected format
            const jsonBytes = new TextEncoder().encode(JSON.stringify(event));
            
            return { 
              value: { chunk: { bytes: jsonBytes } }, 
              done: false 
            };
          },

          async return() {
            const session = self.activeSessions.get(sessionId);
            if (session) session.isActive = false;
            return { value: undefined, done: true };
          },

          async throw(error) {
            const session = self.activeSessions.get(sessionId);
            if (session) session.isActive = false;
            throw error;
          }
        };
      }
    };
  }

  // ---- Bidirectional Streaming ----

  /**
   * Start the bidirectional stream with Bedrock
   */
  async initiateBidirectionalStreaming(sessionId) {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`Stream session ${sessionId} not found`);
    }

    try {
      const asyncIterable = this.createSessionAsyncIterable(sessionId);

      const response = await this.bedrockRuntimeClient.send(
        new InvokeModelWithBidirectionalStreamCommand({
          modelId: NOVA_SONIC_MODEL_ID,
          body: asyncIterable,
        })
      );

      await this.processResponseStream(sessionId, response);

    } catch (error) {
      console.error(`[Nova Sonic] Error in session ${sessionId}:`, error);
      this.dispatchEvent(sessionId, 'error', {
        source: 'bidirectionalStream',
        message: 'Error in bidirectional stream',
        details: error instanceof Error ? error.message : String(error),
      });

      if (session.isActive) {
        this.forceCloseSession(sessionId);
      }
    }
  }

  /**
   * Process the response stream from AWS Bedrock
   */
  async processResponseStream(sessionId, response) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    try {
      for await (const event of response.body) {
        if (!session.isActive) {
          break;
        }

        if (event.chunk?.bytes) {
          try {
            this.updateSessionActivity(sessionId);
            const textResponse = new TextDecoder().decode(event.chunk.bytes);

            try {
              const jsonResponse = JSON.parse(textResponse);

              if (jsonResponse.event?.contentStart) {
                this.dispatchEvent(sessionId, 'contentStart', jsonResponse.event.contentStart);
              } else if (jsonResponse.event?.textOutput) {
                // Check for barge-in indicator
                const textContent = jsonResponse.event.textOutput.content || '';
                if (textContent.includes('{ "interrupted" : true }') || textContent.includes('{"interrupted":true}')) {
                  this.dispatchEvent(sessionId, 'bargeIn', { interrupted: true });
                }
                this.dispatchEvent(sessionId, 'textOutput', jsonResponse.event.textOutput);
              } else if (jsonResponse.event?.audioOutput) {
                this.dispatchEvent(sessionId, 'audioOutput', jsonResponse.event.audioOutput);
              } else if (jsonResponse.event?.toolUse) {
                this.dispatchEvent(sessionId, 'toolUse', jsonResponse.event.toolUse);
                session.toolUseContent = jsonResponse.event.toolUse;
                session.toolUseId = jsonResponse.event.toolUse.toolUseId;
                session.toolName = jsonResponse.event.toolUse.toolName;
              } else if (jsonResponse.event?.contentEnd) {
                if (jsonResponse.event.contentEnd?.type === 'TOOL') {
                  this.dispatchEvent(sessionId, 'toolEnd', {
                    toolUseContent: session.toolUseContent,
                    toolUseId: session.toolUseId,
                    toolName: session.toolName,
                  });
                }
                this.dispatchEvent(sessionId, 'contentEnd', jsonResponse.event.contentEnd);
              } else if (jsonResponse.event?.completionStart) {
                this.dispatchEvent(sessionId, 'completionStart', jsonResponse.event.completionStart);
              } else {
                // Handle other events
                const eventKeys = Object.keys(jsonResponse.event || {});
                if (eventKeys.length > 0) {
                  this.dispatchEvent(sessionId, eventKeys[0], jsonResponse.event);
                }
              }
            } catch (parseErr) {
              // Non-JSON response, skip
            }
          } catch (e) {
            console.error(`[Nova Sonic] Error processing response chunk for session ${sessionId}:`, e);
          }
        } else if (event.modelStreamErrorException) {
          console.error(`[Nova Sonic] Model stream error for session ${sessionId}:`, event.modelStreamErrorException);
          this.dispatchEvent(sessionId, 'error', {
            type: 'modelStreamErrorException',
            source: 'responseStream',
            details: event.modelStreamErrorException?.message || JSON.stringify(event.modelStreamErrorException),
          });
        } else if (event.internalServerException) {
          console.error(`[Nova Sonic] Internal server error for session ${sessionId}:`, event.internalServerException);
          this.dispatchEvent(sessionId, 'error', {
            type: 'internalServerException',
            source: 'responseStream',
            details: event.internalServerException?.message || JSON.stringify(event.internalServerException),
          });
        }
      }

      this.dispatchEvent(sessionId, 'streamComplete', {
        timestamp: new Date().toISOString(),
      });

    } catch (error) {
      console.error(`[Nova Sonic] Error processing response stream for session ${sessionId}:`, error);
      let errorDetails;
      if (error instanceof Error) {
        errorDetails = error.message;
      } else if (error && typeof error === 'object' && error.message) {
        errorDetails = String(error.message);
      } else {
        errorDetails = JSON.stringify(error);
      }
      this.dispatchEvent(sessionId, 'error', {
        source: 'responseStream',
        message: 'Error processing response stream',
        details: errorDetails,
      });
    }
  }

  // ---- Event Setup Methods ----

  /**
   * Setup session start event with inference configuration
   */
  setupSessionStartEvent(sessionId) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const sessionStartEvent = {
      event: {
        sessionStart: {
          inferenceConfiguration: session.inferenceConfig,
        },
      },
    };

    // Add turn detection if configured
    if (session.turnDetectionConfig?.endpointingSensitivity) {
      sessionStartEvent.event.sessionStart.turnDetectionConfiguration = {
        endpointingSensitivity: session.turnDetectionConfig.endpointingSensitivity,
      };
    }

    console.log(`[Nova Sonic] Session start event for ${sessionId}`);
    this.addEventToSessionQueue(sessionId, sessionStartEvent);
  }

  /**
   * Setup prompt start event with audio output configuration
   */
  setupPromptStartEvent(sessionId, voiceId, outputSampleRate = 24000) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const audioOutputConfig = {
      mediaType: 'audio/lpcm',
      sampleRateHertz: outputSampleRate,
      sampleSizeBits: 16,
      channelCount: 1,
      voiceId: voiceId || DefaultAudioOutputConfiguration.voiceId,
      encoding: 'base64',
      audioType: 'SPEECH',
    };

    const promptStartEvent = {
      event: {
        promptStart: {
          promptName: session.promptName,
          textOutputConfiguration: {
            mediaType: 'text/plain',
          },
          audioOutputConfiguration: audioOutputConfig,
        },
      },
    };

    this.addEventToSessionQueue(sessionId, promptStartEvent);
    session.isPromptStartSent = true;
  }

  /**
   * Setup system prompt events (contentStart + textInput + contentEnd)
   */
  setupSystemPromptEvent(sessionId, textConfig = DefaultTextConfiguration, systemPromptContent) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const textPromptID = randomUUID();

    // Content start for system prompt
    const contentStartEvent = {
      event: {
        contentStart: {
          promptName: session.promptName,
          contentName: textPromptID,
          type: 'TEXT',
          interactive: true,
          role: 'SYSTEM',
          textInputConfiguration: textConfig,
        },
      },
    };
    this.addEventToSessionQueue(sessionId, contentStartEvent);

    // Text input content
    const textInputEvent = {
      event: {
        textInput: {
          promptName: session.promptName,
          contentName: textPromptID,
          content: systemPromptContent,
        },
      },
    };
    this.addEventToSessionQueue(sessionId, textInputEvent);

    // Content end for system prompt
    const contentEndEvent = {
      event: {
        contentEnd: {
          promptName: session.promptName,
          contentName: textPromptID,
        },
      },
    };
    this.addEventToSessionQueue(sessionId, contentEndEvent);
  }

  /**
   * Setup audio start event (audio contentStart for user input)
   */
  setupStartAudioEvent(sessionId, audioConfig = DefaultAudioInputConfiguration) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const audioContentStartEvent = {
      event: {
        contentStart: {
          promptName: session.promptName,
          contentName: session.audioContentId,
          type: 'AUDIO',
          interactive: true,
          role: 'USER',
          audioInputConfiguration: audioConfig,
        },
      },
    };

    this.addEventToSessionQueue(sessionId, audioContentStartEvent);
    session.isAudioContentStartSent = true;
  }

  // ---- Audio Streaming ----

  /**
   * Stream an audio chunk for a session
   */
  async streamAudioChunk(sessionId, audioData) {
    const session = this.activeSessions.get(sessionId);
    if (!session || !session.isActive || !session.audioContentId) {
      // Silently drop audio for dead/inactive sessions instead of throwing
      return;
    }

    const base64Data = Buffer.isBuffer(audioData) 
      ? audioData.toString('base64')
      : Buffer.from(audioData).toString('base64');

    this.addEventToSessionQueue(sessionId, {
      event: {
        audioInput: {
          promptName: session.promptName,
          contentName: session.audioContentId,
          content: base64Data,
        },
      },
    });
  }

  // ---- Session Teardown ----

  /**
   * Send content end event
   */
  async sendContentEnd(sessionId) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    this.addEventToSessionQueue(sessionId, {
      event: {
        contentEnd: {
          promptName: session.promptName,
          contentName: session.audioContentId,
        },
      },
    });

    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  /**
   * Send prompt end event
   */
  async sendPromptEnd(sessionId) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    this.addEventToSessionQueue(sessionId, {
      event: {
        promptEnd: {
          promptName: session.promptName,
        },
      },
    });

    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  /**
   * Send session end event and clean up
   */
  async sendSessionEnd(sessionId) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    this.addEventToSessionQueue(sessionId, {
      event: {
        sessionEnd: {},
      },
    });

    // Wait for it to be processed
    await new Promise((resolve) => setTimeout(resolve, 300));

    // Clean up
    session.isActive = false;
    session.closeRequested = true;
    if (session.queueResolve) {
      session.queueResolve();
    }
    this.activeSessions.delete(sessionId);
    this.sessionLastActivity.delete(sessionId);
  }

  /**
   * Close session gracefully
   */
  async closeSession(sessionId) {
    if (this.sessionCleanupInProgress.has(sessionId)) {
      return;
    }
    this.sessionCleanupInProgress.add(sessionId);
    try {
      await this.sendContentEnd(sessionId);
      await this.sendPromptEnd(sessionId);
      await this.sendSessionEnd(sessionId);
    } catch (error) {
      console.error(`[Nova Sonic] Error during closing sequence for session ${sessionId}:`, error);
      const session = this.activeSessions.get(sessionId);
      if (session) {
        session.isActive = false;
        this.activeSessions.delete(sessionId);
        this.sessionLastActivity.delete(sessionId);
      }
    } finally {
      this.sessionCleanupInProgress.delete(sessionId);
    }
  }

  /**
   * Force close a session without graceful shutdown
   */
  forceCloseSession(sessionId) {
    const session = this.activeSessions.get(sessionId);
    if (session) {
      session.isActive = false;
      session.closeRequested = true;
      if (session.queueResolve) {
        session.queueResolve();
      }
      this.activeSessions.delete(sessionId);
      this.sessionLastActivity.delete(sessionId);
    }
  }

  // ---- Event Handling ----

  /**
   * Register an event handler for a session
   */
  registerEventHandler(sessionId, eventType, handler) {
    const session = this.activeSessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    session.responseHandlers.set(eventType, handler);
  }

  /**
   * Dispatch an event to registered handlers
   */
  dispatchEvent(sessionId, eventType, data) {
    const session = this.activeSessions.get(sessionId);
    if (!session) return;

    const handler = session.responseHandlers.get(eventType);
    if (handler) {
      try {
        handler(data);
      } catch (e) {
        console.error(`[Nova Sonic] Error in ${eventType} handler for session ${sessionId}:`, e);
      }
    }

    // Also dispatch to "any" handlers
    const anyHandler = session.responseHandlers.get('any');
    if (anyHandler) {
      try {
        anyHandler({ type: eventType, data });
      } catch (e) {
        console.error(`[Nova Sonic] Error in 'any' handler for session ${sessionId}:`, e);
      }
    }
  }
}

module.exports = {
  NovaSonicBidirectionalStreamClient,
  StreamSession,
  DefaultAudioInputConfiguration,
  DefaultAudioOutputConfiguration,
  DefaultTextConfiguration,
  NOVA_SONIC_MODEL_ID,
};
