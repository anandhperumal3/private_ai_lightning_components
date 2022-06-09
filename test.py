import os
from private_ai_synthetic_data_generator import PrivateAISyntheticData

def send_message():

    pai = PrivateAISyntheticData(
        key= 'INTERNAL_TESTING_UNLIMITED_REALLY',
        mode='standard',
        text_feature='text',
        output_path='./test.csv',
        host='localhost',
        port='8080'
    )
    assert  isinstance(pai.run(input_text_or_path='Hi'), str)
    # assert pai.run(input_text_or_path='', action='', output_dir='') == None
send_message()