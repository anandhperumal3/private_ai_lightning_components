import os
from privateAI_synthetic_data_generator import PrivateAISyntheticData



def test_send_message():
    pai_access_token = os.environ['PrivateAI_key']
    #telegram_chatid = os.environ['TEST_TELEGRAM_CHAT_ID']

    pai = PrivateAISyntheticData(
        key= pai_access_token,
        mode='standard',
        text_features='review',
        url = "http://localhost:8080/deidentify_text"
    )

    assert  isinstance(pai.run(input_text_or_path='', action='', output_dir=''), str)
    assert pai.run(input_text_or_path='', action='', output_dir='') == None
