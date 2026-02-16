/**
 * Nova Sonic Client - Voice AI Integration
 * Adapted from V1 public/src/main.js for use in Svelte frontend
 */

import { AudioPlayer } from './lib/play/AudioPlayer.js';
import { ChatHistoryManager } from './lib/util/ChatHistoryManager.js';

export class NovaSonicClient {
    constructor(options = {}) {
        this.socket = null;
        this.audioPlayer = new AudioPlayer();
        this.audioContext = null;
        this.audioStream = null;
        this.isStreaming = false;
        this.processor = null;
        this.sourceNode = null;
        this.analyser = null;
        this.sessionInitialized = false;
        this.manualDisconnect = false;
        
        // Callbacks
        this.onStatusChange = options.onStatusChange || (() => {});
        this.onTextOutput = options.onTextOutput || (() => {});
        this.onError = options.onError || (() => {});
        this.onSessionStart = options.onSessionStart || (() => {});
        this.onSessionEnd = options.onSessionEnd || (() => {});
        this.onAudioLevel = options.onAudioLevel || (() => {});
        this.onContentStart = options.onContentStart || (() => {});
        this.onContentEnd = options.onContentEnd || (() => {});
        
        // Audio config
        this.TARGET_SAMPLE_RATE = 16000;
        this.isFirefox = typeof navigator !== 'undefined' && navigator.userAgent.toLowerCase().includes('firefox');
        this.samplingRatio = 1;
        
        // Chat history
        this.chat = { history: [] };
        this.chatRef = { current: this.chat };
        this.chatHistoryManager = null;
        
        // State tracking
        this.role = null;
        this.displayAssistantText = false;
        
        // Audio level monitoring
        this.audioLevelInterval = null;
    }

    async connect(serverUrl = '') {
        return new Promise((resolve, reject) => {
            // Use socket.io from global (loaded via script tag)
            if (typeof io === 'undefined') {
                reject(new Error('Socket.IO not loaded. Make sure to include the socket.io script.'));
                return;
            }

            this.socket = io(serverUrl, {
                transports: ['websocket', 'polling']
            });

            this.socket.on('connect', () => {
                console.log('[NovaSonic] Connected to server');
                this.onStatusChange('connected', 'Connected to server');
                resolve();
            });

            this.socket.on('disconnect', () => {
                console.log('[NovaSonic] Disconnected from server');
                this.sessionInitialized = false;
                if (this.manualDisconnect) {
                    this.manualDisconnect = false;
                    this.onStatusChange('ready', 'Session ended. Ready for new session.');
                } else {
                    this.onStatusChange('disconnected', 'Disconnected from server');
                }
                this.onSessionEnd();
            });

            this.socket.on('error', (error) => {
                console.error('[NovaSonic] Server error:', error);
                this.onError(error);
                this.onStatusChange('error', error.message || 'Server error');
            });

            // Set up all event handlers
            this.setupEventHandlers();

            this.socket.on('connect_error', (error) => {
                console.error('[NovaSonic] Connection error:', error);
                reject(error);
            });
        });
    }

    setupEventHandlers() {
        this.socket.on('contentStart', (data) => {
            console.log('[NovaSonic] Content start:', data);
            this.role = data.role;
            
            if (data.type === 'TEXT' && data.role === 'ASSISTANT') {
                let isSpeculative = false;
                try {
                    if (data.additionalModelFields) {
                        const additionalFields = JSON.parse(data.additionalModelFields);
                        isSpeculative = additionalFields.generationStage === 'SPECULATIVE';
                        this.displayAssistantText = isSpeculative;
                    }
                } catch (e) {
                    console.error('[NovaSonic] Error parsing additionalModelFields:', e);
                }
            }
            
            this.onContentStart(data);
        });

        this.socket.on('textOutput', (data) => {
            console.log('[NovaSonic] Text output:', data);
            
            if (this.role === 'USER' || (this.role === 'ASSISTANT' && this.displayAssistantText)) {
                this.onTextOutput({
                    role: data.role,
                    content: data.content
                });
                
                // Update chat history
                if (this.chatHistoryManager) {
                    this.chatHistoryManager.addTextMessage({
                        role: data.role,
                        message: data.content
                    });
                }
            }
        });

        this.socket.on('audioOutput', (data) => {
            if (data.content) {
                try {
                    const audioData = this.base64ToFloat32Array(data.content);
                    this.audioPlayer.playAudio(audioData);
                } catch (error) {
                    console.error('[NovaSonic] Error processing audio:', error);
                }
            }
        });

        this.socket.on('contentEnd', (data) => {
            console.log('[NovaSonic] Content end:', data);
            
            if (data.stopReason) {
                if (data.stopReason.toUpperCase() === 'END_TURN') {
                    if (this.chatHistoryManager) {
                        this.chatHistoryManager.endTurn();
                    }
                } else if (data.stopReason.toUpperCase() === 'INTERRUPTED') {
                    console.log('[NovaSonic] Interrupted by user');
                    this.audioPlayer.bargeIn();
                }
            }
            
            this.onContentEnd(data);
        });

        this.socket.on('streamComplete', () => {
            console.log('[NovaSonic] Stream complete');
            if (this.isStreaming) {
                this.stopStreaming();
            }
            this.onStatusChange('ready', 'Session complete. Ready for new session.');
        });

        this.socket.on('audioReady', () => {
            console.log('[NovaSonic] Audio ready');
            this.onStatusChange('recording', 'Recording...');
        });
    }

    async initAudio() {
        try {
            this.onStatusChange('initializing', 'Requesting microphone access...');

            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });

            if (this.isFirefox) {
                this.audioContext = new AudioContext();
            } else {
                this.audioContext = new AudioContext({
                    sampleRate: this.TARGET_SAMPLE_RATE
                });
            }

            this.samplingRatio = this.audioContext.sampleRate / this.TARGET_SAMPLE_RATE;

            // Create analyser for audio level display
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;

            await this.audioPlayer.start();

            // Initialize chat history manager
            this.chatHistoryManager = ChatHistoryManager.getInstance(
                this.chatRef,
                (newChat) => {
                    this.chat = { ...newChat };
                    this.chatRef.current = this.chat;
                }
            );

            this.onStatusChange('ready', 'Microphone ready. Click to start.');
            return true;
        } catch (error) {
            console.error('[NovaSonic] Error accessing microphone:', error);
            this.onError(error);
            this.onStatusChange('error', 'Microphone error: ' + error.message);
            return false;
        }
    }

    async initializeSession(systemPrompt) {
        if (this.sessionInitialized) return;

        this.onStatusChange('initializing', 'Initializing session...');

        try {
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject(new Error('Connection timeout')), 5000);

                this.socket.emit('initializeConnection', (ack) => {
                    clearTimeout(timeout);
                    if (ack?.success) resolve();
                    else reject(new Error(ack?.error || 'Connection failed'));
                });
            });

            // Send events in sequence
            this.socket.emit('promptStart');
            this.socket.emit('systemPrompt', systemPrompt);
            this.socket.emit('audioStart');

            this.sessionInitialized = true;
            this.onStatusChange('ready', 'Session initialized');
        } catch (error) {
            console.error('[NovaSonic] Failed to initialize session:', error);
            this.onError(error);
            this.onStatusChange('error', 'Session initialization failed');
            throw error;
        }
    }

    async startStreaming(systemPrompt) {
        if (this.isStreaming) return;

        try {
            // Reconnect if disconnected
            if (!this.socket?.connected) {
                await this.connect();
            }

            // Restart audioPlayer if needed
            if (!this.audioPlayer.initialized) {
                await this.audioPlayer.start();
            }

            // Initialize session if needed
            if (!this.sessionInitialized) {
                await this.initializeSession(systemPrompt);
            }

            this.onSessionStart();

            // Create audio processor
            this.sourceNode = this.audioContext.createMediaStreamSource(this.audioStream);
            this.sourceNode.connect(this.analyser);

            // Start audio level monitoring
            this.audioLevelInterval = setInterval(() => {
                const level = this.getAudioLevel();
                this.onAudioLevel(level);
            }, 50);

            // Create script processor for audio capture
            if (this.audioContext.createScriptProcessor) {
                this.processor = this.audioContext.createScriptProcessor(512, 1, 1);

                this.processor.onaudioprocess = (e) => {
                    if (!this.isStreaming) return;

                    const inputData = e.inputBuffer.getChannelData(0);
                    const numSamples = Math.round(inputData.length / this.samplingRatio);
                    const pcmData = this.isFirefox
                        ? new Int16Array(numSamples)
                        : new Int16Array(inputData.length);

                    if (this.isFirefox) {
                        for (let i = 0; i < numSamples; i++) {
                            pcmData[i] = Math.max(-1, Math.min(1, inputData[Math.floor(i * this.samplingRatio)])) * 0x7FFF;
                        }
                    } else {
                        for (let i = 0; i < inputData.length; i++) {
                            pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 0x7FFF;
                        }
                    }

                    const base64Data = this.arrayBufferToBase64(pcmData.buffer);
                    this.socket.emit('audioInput', base64Data);
                };

                this.sourceNode.connect(this.processor);
                this.processor.connect(this.audioContext.destination);
            }

            this.isStreaming = true;
            this.onStatusChange('recording', 'Recording...');

        } catch (error) {
            console.error('[NovaSonic] Error starting streaming:', error);
            this.onError(error);
            this.onStatusChange('error', 'Failed to start: ' + error.message);
        }
    }

    stopStreaming() {
        if (!this.isStreaming) return;

        this.isStreaming = false;

        // Clean up audio processing
        if (this.processor) {
            this.processor.disconnect();
        }
        if (this.sourceNode) {
            this.sourceNode.disconnect();
        }

        // Stop audio level monitoring
        if (this.audioLevelInterval) {
            clearInterval(this.audioLevelInterval);
            this.audioLevelInterval = null;
        }

        this.audioPlayer.bargeIn();

        // Tell server to finalize
        this.socket.emit('stopAudio');

        // End turn in chat history
        if (this.chatHistoryManager) {
            this.chatHistoryManager.endTurn();
        }

        // Reset session
        this.sessionInitialized = false;
        this.manualDisconnect = true;

        // Disconnect
        this.socket.disconnect();

        this.onStatusChange('ready', 'Session ended. Ready for new session.');
        this.onSessionEnd();
    }

    getAudioLevel() {
        if (!this.analyser) return 0;

        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteTimeDomainData(dataArray);

        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            const value = (dataArray[i] - 128) / 128;
            sum += value * value;
        }
        return Math.sqrt(sum / bufferLength);
    }

    // Helper: ArrayBuffer to base64
    arrayBufferToBase64(buffer) {
        const binary = [];
        const bytes = new Uint8Array(buffer);
        for (let i = 0; i < bytes.byteLength; i++) {
            binary.push(String.fromCharCode(bytes[i]));
        }
        return btoa(binary.join(''));
    }

    // Helper: base64 to Float32Array
    base64ToFloat32Array(base64String) {
        const binaryString = window.atob(base64String);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        const int16Array = new Int16Array(bytes.buffer);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0;
        }

        return float32Array;
    }

    disconnect() {
        this.stopStreaming();
        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        this.audioPlayer.stop();
    }

    getChatHistory() {
        return this.chat.history;
    }
}

// Export for use as ES module
export default NovaSonicClient;
