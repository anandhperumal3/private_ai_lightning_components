from privateAI_synthetic_data_generator import PrivateAISyntheticData
import lightning as L
import os

class PrivateAIApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        pai_access_token = os.environ['PrivateAI_key']
        self.pai_sythetic_data_generator = PrivateAISyntheticData(
            key=pai_access_token,
            mode='standard',
            text_features='review',
            url="http://localhost:8080/deidentify_text"
        )

    def run(self):
        self.pai_sythetic_data_generator.synthetic_text('I work at Private AI and I live in Vancouver.')

app = L.LightningApp(PrivateAIApp())
