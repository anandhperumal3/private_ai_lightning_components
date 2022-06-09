from private_ai_synthetic_data_generator import PrivateAISyntheticData
import lightning as L
import os

class PrivateAIApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        pai_access_token = 'INTERNAL_TESTING_UNLIMITED_REALLY'
        self.pai_sythetic_data_generator = PrivateAISyntheticData(
            key=pai_access_token,
            mode='standard',
            output_path='./',
            host='localhost',
            port='8080'
        )
        self.pai_sythetic_data_generator.server_started = True

    def run(self):
        self.pai_sythetic_data_generator.run('I work at Private AI and I live in Vancouver.')

app = L.LightningApp(PrivateAIApp())