<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/lai.png" width="200px">

A Lightning component to make your data private using Private AI

______________________________________________________________________

</div>

## Install

Use these instructions to install:

```bash
git clone https://github.com/anandhperumal3/private_ai_lightning_components.git
cd private_ai_lightning_components
pip install -r requirements.txt
pip install -e .
```

## Use the component

To try the component, copy the code below, save it in `app.py` file and run using `lightning run app app.py`.

```python
import lightning as L
from lightning.storage import Drive
from lightning.frontend import StreamlitFrontend

from private_ai_synthetic_data_generator import PrivateAISyntheticData

import os


class DatasetWork(L.LightningWork):
    def __init__(self) -> None:
        super().__init__(cache_calls=True)
        self.drive = Drive("lit://private_ai_dataset")

    def run(self, input_path, action="put"):
        if action == "put":
            self.drive.put(input_path)
            os.remove(input_path)
        else:
            self.drive.get(input_path)


class Visualizer(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.input_path = None
        self.output_path = None

    def run(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def configure_layout(self):
        return StreamlitFrontend(render_fn=render_fn)


def render_fn(state):
    import streamlit as st
    import pandas as pd

    if not state.input_path or not state.output_path:
        st.title("Working on privatizing the dataset for you!")
        return
    df_input = pd.read_csv(state.input_path).head()
    df_output = pd.read_csv(state.output_path).drop(['Unnamed: 0'],axis=1).head()
    st.write("Input Dataset")
    st.table(df_input)
    st.write("Output Dataset")
    st.table(df_output)


class PrivateAIApp(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        pai_access_token = 'INTERNAL_TESTING_UNLIMITED_REALLY'
        self.dataset_work = DatasetWork()
        self.output_path = "./output.csv"
        self.pai_sythetic_data_generator = PrivateAISyntheticData(
            key=pai_access_token,
            mode='standard',
            text_features=["name", "city"],
            host='localhost',
            port=8080,
            drive=self.dataset_work.drive,
            output_path=self.output_path,
        )
        self.input_path = None
        self.visualizer = Visualizer()
        # self.pai_sythetic_data_generator.server_started = True

    def run(self):
        # This will be the path to your input dataset
        self.input_path = "assets/data.csv"

        self.dataset_work.run(self.input_path)
        self.pai_sythetic_data_generator.run(
            input_path=self.input_path
        )
        self.dataset_work.run(self.output_path, action="get")
        self.visualizer.run(self.input_path, self.output_path)

    def configure_layout(self):
        return {"name": "Private AI Synthetic Data Generator", "content": self.visualizer}


app = L.LightningApp(PrivateAIApp())
```
