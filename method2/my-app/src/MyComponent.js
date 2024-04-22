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

// Define Button component
const Button = ({ onClick, children }) => {
  return <button onClick={onClick}>{children}</button>;
};

export default function Component() {
  const [inputSentenceForTraining, setInputSentenceForTraining] = useState('');
  const [selectedLabelForTraining, setSelectedLabelForTraining] = useState('');
  const [inputSentenceForInference, setInputSentenceForInference] = useState('');
  const [selectedLabelForReTraining, setSelectedLabelForReTraining] = useState('');

  const [outputSentiment, setOutputSentiment] = useState('');

  // const runInference = async () => { // inference 
  //   try {
  //     const response = await axios.post(`${BASE_URL}/inference`, { sentence: inputSentenceForInference });
  //     setOutputSentiment(response.data.sentiment);
  //   } catch (error) {
  //     console.error('Error analyzing sentiment:', error);
  //   }
  // };



  const runInference = async () => {
    try {
      if (!inputSentenceForInference.trim()) {
        console.error('Error: Empty sentence provided');
        return; // Or display an error message to the user
      }
      
      // Prepare the sentence for sending (e.g., trim whitespaces)
      const trimmedSentence = inputSentenceForInference.trim();
      
      // const response = await axios.post(`${BASE_URL}/inference`, { sentence: trimmedSentence });
      const response = await axios.post(`${BASE_URL}/inference?sentence=${encodeURIComponent(trimmedSentence)}`);

      setOutputSentiment(response.data.sentiment);
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
    }
  };


  const writeToFolder = async (sentence, label, section) => { // write the sentences to corresponding label folder file
    try {
      // await axios.post(`${BASE_URL}/write_to_folder`, { sentence, label });
      await axios.post(`${BASE_URL}/write_to_folder?sentence=${encodeURIComponent(sentence)}&label=${encodeURIComponent(label)}`);

      if (section === 'training') {
        setInputSentenceForTraining(''); // Clear input field after writing to folder
        setSelectedLabelForTraining('');
      } else if (section === 'retrain') {
        setInputSentenceForInference('');
        setSelectedLabelForReTraining('');
      }
    } catch (error) {
      console.error('Error writing sentence to folder:', error);
    }
  };

  const trainModel = async () => { // train the model from scratch
    try {
      await axios.post(`${BASE_URL}/train_model`);
      console.log('Model training initiated...');
    } catch (error) {
      console.error('Error training model:', error);
    }
  };

  const reTrainModel = async () => {
    
    trainModel(); // Train model with new data
    
  };
  

  return (
    <div className="grid gap-8 lg:grid-cols-2 items-start space-y-8">
      <div className="space-y-4">
        <div className="space-y-2">
          <h2 className="text-3xl font-extrabold tracking-tight">Training</h2>
          <p className="text-gray-500 dark:text-gray-400">Train the sentiment analysis model.</p>
        </div>
        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="sentence">Sentence</Label>
            <Input className="w-full max-w-md" id="sentence" placeholder="Enter a sentence" value={inputSentenceForTraining} onChange={(e) => setInputSentenceForTraining(e.target.value)} />
          </div>
          <div className="space-y-2">
            <Label>Label</Label>
            <div className="flex space-x-2">
              <Button onClick={() => { setSelectedLabelForTraining('positive'); writeToFolder(inputSentenceForTraining, 'positive', 'training'); }}>Positive</Button>
              <Button onClick={() => { setSelectedLabelForTraining('negative'); writeToFolder(inputSentenceForTraining, 'negative', 'training'); }}>Negative</Button>
              <Button onClick={() => { setSelectedLabelForTraining('neutral'); writeToFolder(inputSentenceForTraining, 'neutral', 'training'); }}>Neutral</Button>
              <Button onClick={() => { setSelectedLabelForTraining('unlabelled'); writeToFolder(inputSentenceForTraining, 'unlabelled', 'training'); }}>Unlabelled</Button>
            </div>
          </div>
          <Button onClick={trainModel}>Train Model</Button>
        </div>
      </div>
      <div className="space-y-4">
        <div className="space-y-2">
          <h2 className="text-3xl font-extrabold tracking-tight">Inference</h2>
          <p className="text-gray-500 dark:text-gray-400">Predict the sentiment of a sentence.</p>
        </div>
        <div className="space-y-2">
          <Label htmlFor="inference">Sentence</Label>
          <Input id="inference" placeholder="Enter a sentence" value={inputSentenceForInference} onChange={(e) => setInputSentenceForInference(e.target.value)} />
        </div>
        <Button onClick={runInference}>Run Inference</Button>
        <div className="space-y-2">
          <Label>Output</Label>
          <Input id="output" placeholder="Sentiment" value={outputSentiment} readOnly />
        </div>
        <div className="flex space-x-2">
              <Button onClick={() => { setSelectedLabelForReTraining('positive'); writeToFolder(inputSentenceForInference, 'positive', 'retrain'); }}>Positive</Button>
              <Button onClick={() => { setSelectedLabelForReTraining('negative'); writeToFolder(inputSentenceForInference, 'negative', 'retrain'); }}>Negative</Button>
              <Button onClick={() => { setSelectedLabelForReTraining('neutral'); writeToFolder(inputSentenceForInference, 'neutral', 'retrain'); }}>Neutral</Button>
              <Button onClick={() => { setSelectedLabelForReTraining('unlabelled'); writeToFolder(inputSentenceForInference, 'unlabelled', 'retrain'); }}>Unlabelled</Button>
            </div>
        <Button onClick={reTrainModel}>Re-Train Model</Button>
      </div>
    </div>
  );
}
