import json
import lightning as L
import requests
from datasets import load_dataset

class PrivateAISyntheticData(L.LightningWork):
    def __init__(self, key, mode, text_features, url):
        """
        Private-AI Data Module, for synthetic data generation
        :param key: PAI customer Key
        :param text_features: list of text feature names in the dataset that needs a synthetic data generation
        :param mode: synthetic data generation model type
        :param url: docker url

        """
        super().__init__()
        self.key = key
        self.mode = mode
        self.text_features = text_features
        self.url = url

    def pai_docker_call(self, text) -> str:
        """
        This function makes a call to the PAI docker for the synthetic text generation.
        :param text: input text
        :return: synthetic text
        """
        payload = json.dumps({
            "text": text,
            "key": self.key,
            "fake_entity_accuracy_mode": self.mode
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", self.url, headers=headers, data=payload)
        fake_text = response.json()["result_fake"]
        return fake_text

    def synthetic_text(self, example, text_feature_name) -> object:
        """
        This function modifies the text in the example with synthetic text.
        :param example: example containing the text column as a field
        :param text_feature_name: text feature name
        :return: example containing the synthetic text in the text feature field
        """
        example[text_feature_name] = self.pai_docker_call(example[text_feature_name])
        return example

    def run(self, input_text_or_path: str, action: str = 'individual', output_dir: str = None):
        synthetic_data = None
        for text_feature_name in self.text_feature_names:
            if action == 'batch':
                assert output_dir, "output directory parameter is requried in-order to save the csv"
                synthetic_data = load_dataset('csv',data_files=input_text_or_path)
                synthetic_data = synthetic_data.map(lambda e: self.synthetic_text(e, text_feature_name))
                synthetic_data.to_csv(output_dir)
            else:
                synthetic_data = self.synthetic_text(input_text_or_path, text_feature_name)
        return synthetic_data
