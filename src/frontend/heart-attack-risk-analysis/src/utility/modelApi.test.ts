import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import { handleLoadData, handleCleanData, handleExecuteModel } from './apiFunctions'; // Adjust the import path as necessary

describe('API Interaction Functions', () => {
    let mock;

    beforeAll(() => {
        // Initialize axios mock
        mock = new MockAdapter(axios);
    });

    afterEach(() => {
        // Reset mock after each test
        mock.reset();
    });

    test('handleLoadData successfully loads data', async () => {
        const filePath = 'test.csv';
        const mockResponse = { message: 'Data loaded successfully' };
        
        // Mocking the POST request to /load_data
        mock.onPost('/load_data', { file_path: filePath }).reply(200, mockResponse);

        global.alert = jest.fn();  // Mocking alert function

        await handleLoadData(filePath);

        // Expect the alert to have been called with the mock response message
        expect(global.alert).toHaveBeenCalledWith(`Response: ${mockResponse.message}`);
    });

    test('handleCleanData successfully cleans data', async () => {
        const dfName = 'testDF';
        const mockResponse = { message: 'Data cleaned successfully' };

        // Mocking the POST request to /clean_data
        mock.onPost('/clean_data', { df_name: dfName }).reply(200, mockResponse);

        global.alert = jest.fn();  // Mocking alert function

        await handleCleanData(dfName);

        // Expect the alert to have been called with the mock response message
        expect(global.alert).toHaveBeenCalledWith(`Response: ${mockResponse.message}`);
    });

    test('handleExecuteModel successfully executes model', async () => {
        const modelType = 'decision_tree';
        const mockResponse = { message: 'Model executed successfully' };

        // Mocking the POST request to /execute_model
        mock.onPost('/execute_model', { model_name: modelType }).reply(200, mockResponse);

        global.alert = jest.fn();  // Mocking alert function

        await handleExecuteModel(modelType);

        // Expect the alert to have been called with the mock response message
        expect(global.alert).toHaveBeenCalledWith(`Response: ${mockResponse.message}`);
    });

    // Additional tests can be written to simulate and handle failed requests
});
