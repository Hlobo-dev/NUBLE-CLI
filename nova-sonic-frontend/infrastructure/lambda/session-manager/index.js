/**
 * Session Manager Lambda Handler
 * 
 * Handles session lifecycle management:
 * - Create session
 * - Get session info
 * - End session
 * - Get chat history
 */

const { DynamoDBClient, PutItemCommand, GetItemCommand, DeleteItemCommand, QueryCommand, UpdateItemCommand } = require('@aws-sdk/client-dynamodb');
const { LambdaClient, InvokeCommand } = require('@aws-sdk/client-lambda');
const { marshall, unmarshall } = require('@aws-sdk/util-dynamodb');
const { randomUUID } = require('crypto');

const dynamoClient = new DynamoDBClient({ region: process.env.AWS_REGION });
const lambdaClient = new LambdaClient({ region: process.env.AWS_REGION });

const SESSION_TABLE = process.env.SESSION_TABLE;
const CHAT_HISTORY_TABLE = process.env.CHAT_HISTORY_TABLE;
const NOVA_SONIC_FUNCTION = process.env.NOVA_SONIC_FUNCTION;

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, X-Api-Key',
  'Content-Type': 'application/json',
};

/**
 * Lambda Handler
 */
exports.handler = async (event, context) => {
  console.log('[SessionManager] Event:', JSON.stringify(event));

  // Handle CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers: corsHeaders, body: '' };
  }

  const path = event.path || event.rawPath || '/';
  const method = event.httpMethod || event.requestContext?.http?.method || 'POST';
  
  let body = {};
  if (event.body) {
    try {
      body = typeof event.body === 'string' ? JSON.parse(event.body) : event.body;
    } catch (e) {
      return respond(400, { error: 'Invalid JSON body' });
    }
  }

  try {
    // Route handling
    if (path.includes('/session/create') || (path === '/session' && method === 'POST')) {
      return await createSession(body);
    }
    
    if (path.includes('/session/start')) {
      return await startSession(body);
    }
    
    if (path.includes('/session/end')) {
      return await endSession(body);
    }
    
    if (path.includes('/session/history')) {
      return await getHistory(body);
    }
    
    if (path.includes('/session/info')) {
      return await getSessionInfo(body);
    }
    
    if (path.includes('/health')) {
      return respond(200, { status: 'healthy', timestamp: new Date().toISOString() });
    }

    // Default: treat as session action based on body.action
    const { action } = body;
    switch (action) {
      case 'create':
        return await createSession(body);
      case 'start':
        return await startSession(body);
      case 'end':
        return await endSession(body);
      case 'history':
        return await getHistory(body);
      case 'info':
        return await getSessionInfo(body);
      default:
        return respond(400, { error: `Unknown action: ${action}` });
    }

  } catch (error) {
    console.error('[SessionManager] Error:', error);
    return respond(500, { error: error.message });
  }
};

/**
 * Create a new session
 */
async function createSession(body) {
  const { userId, voiceId = 'tiffany', systemPrompt, tools = [], metadata = {} } = body;

  const sessionId = randomUUID();
  const now = Date.now();
  const ttl = Math.floor(now / 1000) + (24 * 60 * 60); // 24 hours

  const session = {
    sessionId,
    userId: userId || 'anonymous',
    voiceId,
    systemPrompt: systemPrompt || getDefaultPrompt(),
    tools,
    metadata,
    status: 'created',
    createdAt: now,
    updatedAt: now,
    ttl,
  };

  // Save to DynamoDB
  await dynamoClient.send(new PutItemCommand({
    TableName: SESSION_TABLE,
    Item: marshall(session),
  }));

  console.log(`[SessionManager] Created session: ${sessionId}`);

  return respond(200, {
    success: true,
    sessionId,
    status: 'created',
    voiceId,
  });
}

/**
 * Start a session (initialize Nova Sonic stream)
 */
async function startSession(body) {
  const { sessionId } = body;

  if (!sessionId) {
    return respond(400, { error: 'sessionId is required' });
  }

  // Get session from DynamoDB
  const result = await dynamoClient.send(new GetItemCommand({
    TableName: SESSION_TABLE,
    Key: marshall({ sessionId }),
  }));

  if (!result.Item) {
    return respond(404, { error: 'Session not found' });
  }

  const session = unmarshall(result.Item);

  // Invoke Nova Sonic Lambda to start stream
  if (NOVA_SONIC_FUNCTION) {
    await lambdaClient.send(new InvokeCommand({
      FunctionName: NOVA_SONIC_FUNCTION,
      InvocationType: 'Event', // Async
      Payload: JSON.stringify({
        action: 'start_session',
        sessionId,
        data: {
          voiceId: session.voiceId,
          systemPrompt: session.systemPrompt,
          tools: session.tools,
        },
      }),
    }));
  }

  // Update session status
  await dynamoClient.send(new UpdateItemCommand({
    TableName: SESSION_TABLE,
    Key: marshall({ sessionId }),
    UpdateExpression: 'SET #status = :status, updatedAt = :now',
    ExpressionAttributeNames: { '#status': 'status' },
    ExpressionAttributeValues: marshall({ ':status': 'active', ':now': Date.now() }),
  }));

  console.log(`[SessionManager] Started session: ${sessionId}`);

  return respond(200, {
    success: true,
    sessionId,
    status: 'active',
  });
}

/**
 * End a session
 */
async function endSession(body) {
  const { sessionId } = body;

  if (!sessionId) {
    return respond(400, { error: 'sessionId is required' });
  }

  // Invoke Nova Sonic Lambda to end stream
  if (NOVA_SONIC_FUNCTION) {
    await lambdaClient.send(new InvokeCommand({
      FunctionName: NOVA_SONIC_FUNCTION,
      InvocationType: 'Event',
      Payload: JSON.stringify({
        action: 'end_session',
        sessionId,
      }),
    }));
  }

  // Update session status
  await dynamoClient.send(new UpdateItemCommand({
    TableName: SESSION_TABLE,
    Key: marshall({ sessionId }),
    UpdateExpression: 'SET #status = :status, updatedAt = :now, endedAt = :now',
    ExpressionAttributeNames: { '#status': 'status' },
    ExpressionAttributeValues: marshall({ ':status': 'ended', ':now': Date.now() }),
  }));

  console.log(`[SessionManager] Ended session: ${sessionId}`);

  return respond(200, {
    success: true,
    sessionId,
    status: 'ended',
  });
}

/**
 * Get chat history for a session
 */
async function getHistory(body) {
  const { sessionId, limit = 100 } = body;

  if (!sessionId) {
    return respond(400, { error: 'sessionId is required' });
  }

  const result = await dynamoClient.send(new QueryCommand({
    TableName: CHAT_HISTORY_TABLE,
    KeyConditionExpression: 'sessionId = :sid',
    ExpressionAttributeValues: marshall({ ':sid': sessionId }),
    ScanIndexForward: true,
    Limit: limit,
  }));

  const history = (result.Items || []).map(item => unmarshall(item));

  return respond(200, {
    success: true,
    sessionId,
    history,
    count: history.length,
  });
}

/**
 * Get session info
 */
async function getSessionInfo(body) {
  const { sessionId } = body;

  if (!sessionId) {
    return respond(400, { error: 'sessionId is required' });
  }

  const result = await dynamoClient.send(new GetItemCommand({
    TableName: SESSION_TABLE,
    Key: marshall({ sessionId }),
  }));

  if (!result.Item) {
    return respond(404, { error: 'Session not found' });
  }

  const session = unmarshall(result.Item);

  return respond(200, {
    success: true,
    session: {
      sessionId: session.sessionId,
      userId: session.userId,
      voiceId: session.voiceId,
      status: session.status,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
    },
  });
}

/**
 * Helper: Create response
 */
function respond(statusCode, body) {
  return {
    statusCode,
    headers: corsHeaders,
    body: JSON.stringify(body),
  };
}

/**
 * Default system prompt
 */
function getDefaultPrompt() {
  return `You are a warm, professional, and helpful AI assistant. Give accurate answers that sound natural, direct, and human.
Start by answering the user's question clearly in 1–2 sentences. Then, expand only enough to make the answer understandable, staying within 3–5 short sentences total.
Avoid sounding like a lecture or essay. Be conversational and friendly.`;
}
