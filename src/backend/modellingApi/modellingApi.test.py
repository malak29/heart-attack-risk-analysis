import unittest
import json
from your_flask_app_file import app  # Import your Flask app

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_load_data(self):
        response = self.app.post('/load_data', data=json.dumps({'file_path': 'path/to/your/test_dataset.csv'}), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('Data loaded successfully', response_data['message'])

    def test_clean_data(self):
        response = self.app.post('/clean_data', data=json.dumps({'file_path': 'path/to/your/test_dataset.csv'}), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('Data cleaned successfully', response_data['message'])

    def test_train_random_forest(self):
        # Prepare sample data for testing
        test_data = {
            # Include sample data here that matches the expected input format for your model
        }
        response = self.app.post('/train_random_forest', data=json.dumps(test_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('Random Forest model trained', response_data['message'])
        self.assertTrue('accuracy' in response_data)

    def test_train_gradient_boosting(self):
        # Prepare sample data for testing
        test_data = {
            # Include sample data here that matches the expected input format for your model
        }
        response = self.app.post('/train_gradient_boosting', data=json.dumps(test_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertIn('Gradient Boosting model trained', response_data['message'])
        self.assertTrue('accuracy' in response_data)

    # Add more test methods as needed for other endpoints

if __name__ == '__main__':
    unittest.main()