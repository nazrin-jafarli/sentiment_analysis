import React, { useState } from 'react';
import axios from 'axios';
import { BASE_URL } from './config';

// Define Label component
const Label = ({ htmlFor, children }) => {
  return <label htmlFor={htmlFor}>{children}</label>;
};

// Define Input component
const Input = ({ id, placeholder, value, onChange }) => {
  return <input id={id} placeholder={placeholder} value={value} onChange={onChange} />;
};

// Define Button component with inline styles for color
const Button = ({ onClick, children }) => {
  return (
    <button 
      onClick={onClick} 
      style={{
        backgroundColor: '#00008B', // Dark green background color
        color: 'white', // White text color
        border: '1px solid #ADD8E6', // Light green border
        borderRadius: '4px', // Rounded corners
        padding: '0.5rem 1rem', // Padding
        cursor: 'pointer', // Cursor on hover
        fontSize: '1rem', // Font size
        fontWeight: 'bold' // Font weight
      }}
    >
      {children}
    </button>
  );
};

export default function Component() {
  const [inputSentenceForTraining, setInputSentenceForTraining] = useState('');
  const [inputSentenceForInference, setInputSentenceForInference] = useState('');
  const [outputSentiment, setOutputSentiment] = useState('');

  const runInference = async () => {
    try {
      if (!inputSentenceForInference.trim()) {
        console.error('Error: Empty sentence provided');
        return; // Or display an error message to the user
      }
      
      const trimmedSentence = inputSentenceForInference.trim();
      const response = await axios.post(`${BASE_URL}/inference?sentence=${encodeURIComponent(trimmedSentence)}`);

      setOutputSentiment(response.data.sentiment);
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
    }
  };

  const writeToFolder = async (sentence, label, section) => {
    try {
      await axios.post(`${BASE_URL}/write_to_folder?sentence=${encodeURIComponent(sentence)}&label=${encodeURIComponent(label)}`);

      if (section === 'training') {
        setInputSentenceForTraining('');
      } else if (section === 'retrain') {
        setInputSentenceForInference('');
      }
    } catch (error) {
      console.error('Error writing sentence to folder:', error);
    }
  };

  const trainModel = async () => {
    try {
      await axios.post(`${BASE_URL}/train_model`);
      console.log('Model training initiated...');
    } catch (error) {
      console.error('Error training model:', error);
    }
  };

  const reTrainModel = async () => {
    trainModel();
  };

  return (
    <div className="container">
      {/* Welcome Text Section */}
      <div className="welcome-text" style={{ textAlign: 'center' }}>
        <h1>Welcome to Automated User Analyzer Tool!</h1>
        <p>Train your model and predict sentiment of sentences.</p>
      </div>
      <div className="training-inference" style={{ display: 'flex' }}>
        {/* Training Section */}
        <div className="w-1/2" style={{ display: 'flex', flexDirection: 'column', marginRight: '200px',marginLeft: '100px' }}>
          <div className="space-y-4">
            <h2 className="text-3xl font-extrabold tracking-tight">Training</h2>
            <p className="text-gray-500 dark:text-gray-400">Train the sentiment analysis model.</p>
            <div className="space-y-2">
              <Label htmlFor="sentence">Sentence</Label>
              <Input
                className="w-full max-w-md"
                id="sentence"
                placeholder="Enter a sentence"
                value={inputSentenceForTraining}
                onChange={(e) => setInputSentenceForTraining(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <div className="flex space-x-2">
                <Button
                  onClick={() => {
                    writeToFolder(inputSentenceForTraining, 'positive', 'training');
                  }}
                >
                  Positive
                </Button>
                <Button
                  onClick={() => {
                    writeToFolder(inputSentenceForTraining, 'negative', 'training');
                  }}
                >
                  Negative
                </Button>
                <Button
                  onClick={() => {
                    writeToFolder(inputSentenceForTraining, 'neutral', 'training');
                  }}
                >
                  Neutral
                </Button>
                <Button
                  onClick={() => {
                    writeToFolder(inputSentenceForTraining, 'unlabelled', 'training');
                  }}
                >
                  Unlabelled
                </Button>
              </div>
            </div>
            <Button onClick={trainModel}>Train Model</Button>
          </div>
        </div>
    
        {/* Inference Section */}
        <div className="w-1/2" style={{ display: 'flex', flexDirection: 'column', marginRight: '20px' }}>
          <div className="space-y-4">
            <h2>Inference</h2>
            <p className="text-gray-500 dark:text-gray-400">Predict the sentiment of a sentence.</p>
            <div className="space-y-2">
              <Label htmlFor="inference">Sentence</Label>
              <Input
                id="inference"
                placeholder="Enter a sentence"
                value={inputSentenceForInference}
                onChange={(e) => setInputSentenceForInference(e.target.value)}
              />
            </div>
            <Button onClick={runInference}>Run Inference</Button>
            <div className="space-y-2">
              <Label>Output</Label>
              <Input id="output" placeholder="Sentiment" value={outputSentiment} readOnly />
            </div>
            <div className="flex space-x-2">
              <Button
                onClick={() => {
                  writeToFolder(inputSentenceForInference, 'positive', 'retrain');
                }}
              >
                Positive
              </Button>
              <Button
                onClick={() => {
                  writeToFolder(inputSentenceForInference, 'negative', 'retrain');
                }}
              >
                Negative
              </Button>
              <Button
                onClick={() => {
                  writeToFolder(inputSentenceForInference, 'neutral', 'retrain');
                }}
              >
                Neutral
              </Button>
              <Button
                onClick={() => {
                  writeToFolder(inputSentenceForInference, 'unlabelled', 'retrain');
                }}
              >
                Unlabelled
              </Button>
            </div>
            <Button onClick={reTrainModel}>Re-Train Model</Button>
          </div>
        </div>
      </div>
    </div>
  );
}
