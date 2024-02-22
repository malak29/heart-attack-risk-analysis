import axios from 'axios';

export const handleLoadData = async (filePath) => {
    try {
        const response = await axios.post('/load_data', { file_path: filePath });
        alert(`Response: ${response.data.message}`);
    } catch (error) {
        console.error('Error loading data:', error);
        alert('Failed to load data');
    }
};

export const handleCleanData = async (dfName) => {
    try {
        // Add more parameters as needed for cleaning
        const response = await axios.post('/clean_data', { df_name: dfName });
        alert(`Response: ${response.data.message}`);
    } catch (error) {
        console.error('Error cleaning data:', error);
        alert('Failed to clean data');
    }
};

export const handleExecuteModel = async (modelType) => {
    try {
        // Add parameters for model execution
        const response = await axios.post('/execute_model', { model_name: modelType });
        alert(`Response: ${response.data.message}`);
    } catch (error) {
        console.error('Error executing model:', error);
        alert('Failed to execute model');
    }
};