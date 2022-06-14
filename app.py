import lightning as L
from lightning.storage import Drive

from private_ai_synthetic_data_generator import PrivateAISyntheticData

import os


class DatasetWork(L.LightningWork):
    def __init__(self) -> None:
        super().__init__(cache_calls=True)
        self.drive = Drive("lit://private_ai_dataset")

    def run(self, input_path):
        self.drive.put(input_path)
        os.remove(input_path)


class PrivateAIApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        pai_access_token = 'INTERNAL_TESTING_UNLIMITED_REALLY'
        self.dataset_work = DatasetWork()
        self.pai_sythetic_data_generator = PrivateAISyntheticData(
            key=pai_access_token,
            mode='standard',
            text_features=["name", "city"],
            host='localhost',
            port=8080,
            drive=self.dataset_work.drive,
            output_path='./output.csv',
        )
        # self.pai_sythetic_data_generator.server_started = True

    def run(self):
        # self.pai_sythetic_data_generator.run('I work at Private AI and I live in Vancouver.')

        # This will be the path to your input dataset
        path = "data.csv" 

        self.dataset_work.run(path)
        # self.pai_sythetic_data_generator.run(
        #     input_path=path,
        #     action="batch",
        # )

        # OR you can just pass the text
        self.pai_sythetic_data_generator.run(
            input_text="I work at Private AI and I live in Vancouver.",
            action='individual'
        )


app = L.LightningApp(PrivateAIApp())
