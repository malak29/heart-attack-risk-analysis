import React, { useState } from 'react';
import axios from 'axios';

const Homepage = () => {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState('');
    const [modelData, setModelData] = useState({});  // State to hold model input data

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    // Handler for loading and preparing data for model training
    const handlePrepareData = async () => {
        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                // Assuming there's an endpoint to prepare and return model data
                const response = await axios.post('/prepare_data', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                });
                setModelData(response.data);  // Save prepared data for model training
                setResult("Data prepared successfully for model training.");
            } catch (error) {
                console.error('Error preparing data:', error);
                setResult('Failed to prepare data for model training.');
            }
        }
    };

    const handleTrainRandomForest = async () => {
        try {
            const response = await axios.post('/train_random_forest', modelData);
            setResult(`Random Forest Model Trained. Accuracy: ${response.data.accuracy}`);
        } catch (error) {
            console.error('Error training Random Forest model:', error);
            setResult('Failed to train Random Forest model.');
        }
    };

    const handleTrainGradientBoosting = async () => {
        try {
            const response = await axios.post('/train_gradient_boosting', modelData);
            setResult(`Gradient Boosting Model Trained. Accuracy: ${response.data.accuracy}`);
        } catch (error) {
            console.error('Error training Gradient Boosting model:', error);
            setResult('Failed to train Gradient Boosting model.');
        }
    };

    // Add more handlers for other models as needed

    return (
        <div>
            <h2>Data Analysis</h2>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handlePrepareData}>Prepare Data</button>
            <button onClick={handleTrainRandomForest}>Train Random Forest</button>
            <button onClick={handleTrainGradientBoosting}>Train Gradient Boosting</button>
            {/* Add more buttons for other models */}

            <div>
                <h3>Result:</h3>
                <p>{result}</p>
            </div>
        </div>
    );
};

export default Homepage;
