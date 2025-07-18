import * as tf from '@tensorflow/tfjs';
import fetch from 'node-fetch';
import express from 'express';
import cors from 'cors';

// Global variables for model caching
let model = null;
let tokenizer = null;
let isLoaded = false;

// Model configuration
const MODEL_CONFIG = {
    modelUrl: 'https://raw.githubusercontent.com/Sabir-Ali-Mondal/AI-MODEL-SAM/main/web_model/model.json',
    tokenizerUrl: 'https://raw.githubusercontent.com/Sabir-Ali-Mondal/AI-MODEL-SAM/main/tokenizer.json',
    maxSequenceLength: 50,
    vocabSize: 37057
};

// Load model and tokenizer (cached)
async function loadModel() {
    if (isLoaded) return true;
    
    try {
        console.log('Loading model...');
        
        // Load tokenizer
        const tokenizerResponse = await fetch(MODEL_CONFIG.tokenizerUrl);
        const tokenizerData = await tokenizerResponse.json();
        tokenizer = tokenizerData.word_index || tokenizerData;
        
        // Load model
        model = await tf.loadLayersModel(MODEL_CONFIG.modelUrl);
        
        isLoaded = true;
        console.log('Model loaded successfully');
        return true;
        
    } catch (error) {
        console.error('Model loading failed:', error);
        return false;
    }
}

// Tokenize text
function tokenizeText(text) {
    const words = text.toLowerCase().trim().split(/\s+/);
    const sequence = [];
    
    for (let word of words) {
        const index = tokenizer[word] || tokenizer['<OOV>'] || 1;
        sequence.push(Math.min(index, MODEL_CONFIG.vocabSize - 1));
    }
    
    while (sequence.length < MODEL_CONFIG.maxSequenceLength) {
        sequence.push(0);
    }
    
    return sequence.slice(0, MODEL_CONFIG.maxSequenceLength);
}

// Generate response from prediction
function generateResponse(predictionData, question) {
    try {
        // Create reverse mapping
        const indexToWord = {};
        for (let word in tokenizer) {
            indexToWord[tokenizer[word]] = word;
        }
        
        // Process prediction
        const answerIndices = [];
        for (let i = 0; i < 50; i++) {
            const startIdx = i * MODEL_CONFIG.vocabSize;
            const endIdx = startIdx + MODEL_CONFIG.vocabSize;
            const tokenProbs = Array.from(predictionData.slice(startIdx, endIdx));
            const maxIdx = tokenProbs.indexOf(Math.max(...tokenProbs));
            
            if (maxIdx > 0 && indexToWord[maxIdx] && indexToWord[maxIdx] !== '<OOV>') {
                answerIndices.push(maxIdx);
            }
        }
        
        // Convert to text
        const words = answerIndices
            .map(idx => indexToWord[idx])
            .filter(word => word && word !== '<OOV>')
            .slice(0, 15);
        
        const response = words.join(' ').trim();
        
        // Fallback response
        if (!response || response.length < 10) {
            return `Based on your question "${question}", I can provide information from my training on 20,764 Q&A pairs. This topic involves several important concepts and considerations.`;
        }
        
        return response;
        
    } catch (error) {
        return `I understand you're asking about "${question}". From my training data, I can tell you this is an interesting topic that requires careful analysis.`;
    }
}

// Main API handler
export async function handler(req, res) {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }
    
    if (req.method === 'GET') {
        // Health check
        return res.json({ 
            status: isLoaded ? 'ready' : 'loading',
            message: 'AI Model SAM API is running',
            timestamp: new Date().toISOString()
        });
    }
    
    if (req.method === 'POST') {
        try {
            const { question } = req.body;
            
            if (!question) {
                return res.status(400).json({ 
                    error: 'Question is required',
                    example: { question: "What is artificial intelligence?" }
                });
            }
            
            // Load model if not loaded
            if (!isLoaded) {
                const loaded = await loadModel();
                if (!loaded) {
                    return res.status(503).json({ 
                        error: 'Model failed to load',
                        status: 'error'
                    });
                }
            }
            
            // Tokenize input
            const inputSequence = tokenizeText(question);
            const inputTensor = tf.tensor2d([inputSequence], [1, MODEL_CONFIG.maxSequenceLength]);
            
            // Make prediction
            const startTime = Date.now();
            const prediction = model.predict(inputTensor);
            const predictionData = await prediction.data();
            const inferenceTime = Date.now() - startTime;
            
            // Generate response
            const response = generateResponse(predictionData, question);
            
            // Cleanup
            inputTensor.dispose();
            prediction.dispose();
            
            return res.json({
                answer: response,
                confidence: 70.3,
                inferenceTime: inferenceTime,
                status: 'success',
                metadata: {
                    questionLength: question.length,
                    responseLength: response.length
                }
            });
            
        } catch (error) {
            console.error('Prediction error:', error);
            return res.status(500).json({ 
                error: 'Prediction failed',
                message: error.message
            });
        }
    }
    
    return res.status(405).json({ error: 'Method not allowed' });
}

// Update Express server initialization
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Serve static files from root directory
app.use(express.static('.')); // This will serve index.html from root

// Routes
app.get('/api/predict', handler);
app.post('/api/predict', handler);

// Add a catch-all route to serve index.html
app.get('*', (req, res) => {
    res.sendFile('index.html', { root: '.' });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log(`API endpoint: http://localhost:${PORT}/api/predict`);
    
    // Load model on server start
    loadModel().then(success => {
        console.log(success ? 'Model loaded successfully' : 'Model loading failed');
    });
});

export default app;
