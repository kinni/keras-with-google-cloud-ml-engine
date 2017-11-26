# -*- coding: utf-8 -*-

import json
import codecs
from trainer import model

text = u"無冷場，笑點又多，一定要去睇"
# text = u"又話又話好多笑點？邊個話好睇架？一定係打手。"
# text = u"好好笑，心情唔好睇完會即刻笑返。"
# text = u"幾好睇，一啲都唔覺得悶。"

input_sample = {
    'text': model.get_input_data("./data/data.csv", text)[0]
}

print('text:',)
print(text)
print('input sequence: ')
print(input_sample)

with codecs.open("sample.json", 'w', encoding='utf-8') as outfile:
    json.dump(input_sample, outfile)
