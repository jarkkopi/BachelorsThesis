# run_from_cache.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import spacy
from collections import defaultdict
from tabulate import tabulate

# === Paths ===
json_file_path = './data/val_captions.json'
csv_predictions_path = './MTURK/mturk_audio_tags_dynamic.csv'

predictions_df = pd.read_csv(csv_predictions_path)
predictions_df.set_index('filename', inplace=True)

# === Parameters ===
num_clips = 10
top_n_tags = 3

ground_truth_tags_dict = {
    "4963357278.wav": ['Child speech, kid speaking', 'Speech'],
    "4964516093.wav": ['Child speech, kid speaking', 'Children playing', 'Laughter', 'Speech'],
    "4967363921.wav": ['Chatter', 'Female speech, woman speaking', 'Outside, urban or manmade'],
    "4969723020.wav": ['Rustling leaves', 'Speech'],
    "4972231631.wav": ['Dishes, pots, and pans', 'Speech'],
    "4981531978.wav": ['Speech'],
    "10164182964.wav": ['Musical instrument'],
    "10192494165.wav": ['Clip-clop', 'Speech'],
    "10211516755.wav": ['Child speech, kid speaking', 'Conversation', 'Female speech, woman speaking', 'Male speech, man speaking'],
    "10255078225.wav": ['Music', 'Musical instrument', 'Speech'],
    "4983163710.wav": ['Female speech, woman speaking', 'Speech'],
    "10429632423.wav": ['Boat, Water vehicle', 'Ocean', 'Speech', 'Waves, surf', 'Wind', 'Wind noise (microphone)'],
    "4983786005.wav": ['Speech'],
    "10433664864.wav": ['Child speech, kid speaking', 'Speech'],
    "4987887725.wav": ['Carnatic music', 'Child singing', 'Music', 'Wind instrument, woodwind instrument'],
    "10440439946.wav": ['Child speech, kid speaking', 'Children playing'],
    "10522689523.wav": ['Inside, small room', 'Rain', 'Silence', 'Speech', 'Whispering'],
    "4992338194.wav": ['Cheering', 'Clapping', 'Crowd', 'Music', 'Musical instrument', 'Speech'],
    "10542271055.wav": ['Child speech, kid speaking', 'Speech'],
    "4999127441.wav": ['Clapping', 'Female singing', 'Singing', 'Speech'],
    "10576728026.wav": ['Crowd', 'Music'],
    "4999665957.wav": ['Alarm', 'Child speech, kid speaking', 'Children playing', 'Music', 'Speech'],
    "10612297974.wav": ['Clapping', 'Singing', 'Speech'],
    "5008618500.wav": ['Child speech, kid speaking', 'Speech'],
    "5013421988.wav": ['Speech'],
    "10636949046.wav": ['Child speech, kid speaking', 'Children playing', 'Outside, urban or manmade', 'Speech'],
    "5015000984.wav": ['Boat, Water vehicle', 'Water'],
    "10651643304.wav": ['Guitar', 'Music', 'Musical instrument', 'Singing'],
    "5017166671.wav": ['Guitar', 'Music', 'Musical instrument', 'Plucked string instrument', 'Speech'],
    "10685598604.wav": ['Male speech, man speaking', 'Speech'],
    "5029903979.wav": ['Speech'],
    "10701071663.wav": ['Child speech, kid speaking', 'Children playing', 'Outside, rural or natural', 'Speech', 'Water'],
    "10717276603.wav": ['Air conditioning', 'Music', 'Speech', 'Wind chime'],
    "5033359225.wav": ['Engine', 'Speech'],
    "5041383030.wav": ['Male speech, man speaking', 'Speech'],
    "10735513724.wav": ['Child singing', 'Singing'],
    "5042865125.wav": ['Child speech, kid speaking', 'Speech'],
    "10745999374.wav": ['Belly laugh', 'Child speech, kid speaking', 'Children playing', 'Female speech, woman speaking', 'Music', 'Speech'],
    "5049587564.wav": ['Female speech, woman speaking', 'Hubbub, speech noise, speech babble', 'Speech'],
    "10793023296.wav": ['Hubbub, speech noise, speech babble', 'Keys jangling', 'Speech', 'Vehicle'],
    "5050116958.wav": ['Female speech, woman speaking'],
    "5056964274.wav": ['Female speech, woman speaking', 'Speech'],
    "10815467905.wav": ['Chuckle, chortle', 'Giggle', 'Laughter', 'Speech'],
    "5064681723.wav": ['Speech'],
    "10860342315.wav": ['Baby laughter', 'Child speech, kid speaking', 'Music', 'Speech', 'Synthetic singing'],
    "5070310138.wav": ['Child speech, kid speaking', 'Hands', 'Speech'],
    "10871643364.wav": ['Bouncing', 'Music', 'Speech'],
    "5082198541.wav": ['Child singing'],
    "10981207233.wav": ['Sink (filling or washing)', 'Speech'],
    "5082890299.wav": ['Baby laughter', 'Giggle'],
    "10995087244.wav": ['Child speech, kid speaking', 'Children playing', 'Speech'],
    "5096827532.wav": ['Child singing', 'Child speech, kid speaking', 'Female singing'],
    "11047438904.wav": ['Railroad car, train wagon', 'Speech'],
    "5104115508.wav": ['Music', 'Musical instrument', 'Singing', 'Speech'],
    "11061971625.wav": ['Chatter', 'Child speech, kid speaking', 'Children playing', 'Speech'],
    "5107014713.wav": ['Children shouting', 'Crowd'],
    "11111957393.wav": ['Music'],
    "5107391081.wav": ['Child speech, kid speaking'],
    "11112199204.wav": ['Child speech, kid speaking', 'Inside, small room', 'Speech'],
    "5107595441.wav": [],
    "5112223863.wav": ['Crowd', 'Female singing', 'Hands', 'Singing'],
    "11117946583.wav": ['Speech'],
    "5112579753.wav": ['Babbling', 'Rodents, rats, mice'],
    "11121715464.wav": ['Child speech, kid speaking', 'Music', 'Speech'],
    "5116088152.wav": ['Speech'],
    "11130480243.wav": ['Female singing', 'Music', 'Musical instrument'],
    "5133435424.wav": ['Child speech, kid speaking', 'Laughter', 'Speech'],
    "11162473963.wav": ['Babbling', 'Speech'],
    "5139813648.wav": ['Child speech, kid speaking', 'Female speech, woman speaking', 'Inside, small room'],
    "11183847675.wav": ['Boat, Water vehicle', 'Dog', 'Domestic animals, pets', 'Speech', 'Whimper (dog)'],
    "5144319075.wav": ['Baby laughter', 'Child speech, kid speaking', 'Giggle', 'Laughter', 'Speech'],
    "11220088276.wav": ['Child speech, kid speaking', 'Speech'],
    "5145149014.wav": ['Babbling', 'Child speech, kid speaking', 'Cough', 'Dishes, pots, and pans', 'Speech'],
    "11222803754.wav": ['Laughter', 'Speech'],
    "5145526755.wav": ['Motorcycle', 'Vehicle'],
    "11232317315.wav": ['Music', 'Singing'],
    "5148501793.wav": ['Speech', 'Waterfall', 'Waves, surf', 'Wind'],
    "11261138255.wav": ['Child speech, kid speaking', 'Chuckle, chortle', 'Speech'],
    "5149181269.wav": ['Child singing', 'Child speech, kid speaking', 'Speech'],
    "11301174465.wav": ['Child speech, kid speaking', 'Inside, small room', 'Laughter', 'Speech'],
    "5162562403.wav": ['Dog'],
    "11329995395.wav": ['Child singing', 'Singing'],
    "5165840822.wav": ['Speech'],
    "11402789396.wav": ['Female speech, woman speaking', 'Speech'],
    "5166736368.wav": ['Baby laughter', 'Speech'],
    "11484028616.wav": ['Male speech, man speaking'],
    "5166753093.wav": ['Music', 'Musical instrument'],
    "5173216136.wav": ['Guitar', 'Music', 'Musical instrument', 'Singing'],
    "11510612945.wav": ['Dog', 'Speech'],
    "5177755044.wav": ['Babbling', 'Speech'],
    "11541421563.wav": ['Music', 'Musical instrument', 'Singing', 'Speech'],
    "5178539050.wav": ['Speech'],
    "11566764085.wav": ['Child speech, kid speaking', 'Speech'],
    "11566980553.wav": ['Child speech, kid speaking', 'Speech'],
    "5179297688.wav": ['Child speech, kid speaking', 'Female speech, woman speaking', 'Speech'],
    "11587211476.wav": ['Outside, urban or manmade', 'Speech'],
    "5182578763.wav": ['Female speech, woman speaking', 'Laughter', 'Male speech, man speaking', 'Speech'],
    "11595682706.wav": ['Cat', 'Speech'],
    "5185070620.wav": ['Child speech, kid speaking', 'Inside, small room', 'Speech'],
    "11633816836.wav": ['Music', 'Vehicle'],
    "5192910012.wav": ['Child speech, kid speaking', 'Speech'],
    "11636217844.wav": ['Female speech, woman speaking', 'Speech'],
    "5196883885.wav": ['Babbling', 'Baby cry, infant cry', 'Baby laughter', 'Child speech, kid speaking', 'Laughter', 'Speech'],
    "11647929285.wav": ['Giggle', 'Laughter', 'Male speech, man speaking', 'Speech'],
    "5202829963.wav": ['Clapping', 'Music', 'Singing', 'Speech'],
    "11772124873.wav": ['Singing'],
    "5202975055.wav": ['Guitar', 'Music', 'Singing'],
    "5206737488.wav": ['Conversation', 'Female speech, woman speaking', 'Inside, small room', 'Laughter', 'Speech'],
    "4366402470.wav": ['Speech'],
    "5208873554.wav": ['Rain', 'Rain on surface', 'Speech'],
    "4367056464.wav": ['Child singing', 'Dog', 'Domestic animals, pets', 'Singing'],
    "5209164320.wav": ['Acoustic guitar', 'Chuckle, chortle', 'Guitar', 'Music', 'Musical instrument', 'Plucked string instrument', 'Singing', 'Speech'],
    "4370774286.wav": ['Chatter', 'Inside, large room or hall', 'Inside, public space', 'Speech'],
    "5211051388.wav": ['Child singing', 'Child speech, kid speaking', 'Musical instrument'],
    "4371640303.wav": ['Babbling', 'Baby cry, infant cry', 'Child speech, kid speaking', 'Inside, small room', 'Speech'],
    "5211277413.wav": ['Speech'],
    "4372494989.wav": ['Babbling', 'Child speech, kid speaking', 'Children playing', 'Male speech, man speaking', 'Speech'],
    "5216965406.wav": ['Child speech, kid speaking', 'Children playing', 'Dishes, pots, and pans', 'Inside, small room'],
    "4380772652.wav": ['Child speech, kid speaking', 'Children playing', 'Speech'],
    "5219718493.wav": ['Outside, urban or manmade', 'Speech'],
    "4390527144.wav": ['Child speech, kid speaking', 'Clicking', 'Inside, small room', 'Speech'],
    "5219728337.wav": ['Child speech, kid speaking', 'Female speech, woman speaking'],
    "5223035448.wav": ['Babbling', 'Child speech, kid speaking'],
    "4399688985.wav": ['Speech'],
    "5224502522.wav": ['Music', 'Musical instrument'],
    "4403191162.wav": ['Child speech, kid speaking', 'Children playing', 'Speech'],
    "5224805531.wav": ['Clapping', 'Music', 'Singing', 'Speech'],
    "4405101727.wav": ['Female speech, woman speaking', 'Hubbub, speech noise, speech babble', 'Inside, public space', 'Speech'],
    "5237219621.wav": ['Conversation', 'Female speech, woman speaking', 'Speech'],
    "5254195028.wav": ['Chatter', 'Music', 'Speech'],
    "4407186551.wav": ['Female singing', 'Music'],
    "5254866216.wav": ['Conversation', 'Female speech, woman speaking', 'Male speech, man speaking', 'Speech'],
    "5263195812.wav": ['Male speech, man speaking', 'Speech'],
    "4410543060.wav": ['Child speech, kid speaking', 'Clapping', 'Speech'],
    "5265004457.wav": ['Conversation', 'Female speech, woman speaking', 'Male speech, man speaking', 'Speech'],
    "4410882051.wav": ['Speech'],
    "5265614788.wav": ['Babbling', 'Children playing', 'Dishes, pots, and pans', 'Inside, small room', 'Speech'],
    "4414047871.wav": ['Music', 'Piano', 'Singing'],
    "5267923023.wav": ['Chatter', 'Chuckle, chortle', 'Conversation', 'Female speech, woman speaking', 'Speech'],
    "4437776075.wav": ['Babbling', 'Child speech, kid speaking', 'Laughter', 'Speech'],
    "5273185164.wav": ['Male speech, man speaking', 'Speech'],
    "5278313911.wav": ['Child speech, kid speaking', 'Speech'],
    "4442953867.wav": ['Child speech, kid speaking', 'Crumpling, crinkling', 'Speech'],
    "4443349080.wav": ['Bird', 'Environmental noise', 'Outside, rural or natural', 'Speech'],
    "5284382213.wav": ['Ocean', 'Wind', 'Wind noise (microphone)'],
    "4447314310.wav": ['Speech'],
    "5286758447.wav": ['Female singing', 'Music', 'Singing'],
    "4451729145.wav": ['Speech'],
    "4458565343.wav": ['Boat, Water vehicle', 'Ocean', 'Speech', 'Waves, surf', 'Wind noise (microphone)'],
    "5287251474.wav": ['Speech', 'Television'],
    "4464727975.wav": ['Child speech, kid speaking', 'Speech'],
    "5293233413.wav": ['Inside, small room', 'Speech'],
    "5295188880.wav": ['Domestic animals, pets', 'Speech'],
    "4465235803.wav": ['Speech', 'Television'],
    "5296671076.wav": ['Child singing', 'Child speech, kid speaking', 'Giggle', 'Speech'],
    "4466788724.wav": ['Babbling', 'Child speech, kid speaking', 'Female speech, woman speaking'],
    "5303332305.wav": ['Child speech, kid speaking', 'Speech'],
    "4474236062.wav": ['Bird vocalization, bird call, bird song', 'Speech'],
    "5305581718.wav": ['Baby laughter', 'Speech'],
    "4474456610.wav": ['Music', 'Speech'],
    "5306545201.wav": ['Guitar', 'Music', 'Musical instrument', 'Strum'],
    "5308844630.wav": ['Rain', 'Rain on surface', 'Speech'],
    "4479407449.wav": ['Female speech, woman speaking'],
    "5309902755.wav": ['Music', 'Musical instrument', 'Violin, fiddle'],
    "4480586020.wav": ['Bathtub (filling or washing)', 'Speech', 'Water'],
    "5310031663.wav": ['Cutlery, silverware', 'Glass', 'Music'],
    "4487616974.wav": ['Babbling', 'Child speech, kid speaking', 'Speech'],
    "5311367704.wav": ['Child speech, kid speaking', 'Inside, small room', 'Speech'],
    "4488066715.wav": ['Child speech, kid speaking', 'Female speech, woman speaking', 'Speech'],
    "5312036289.wav": ['Speech'],
    "4489734073.wav": ['Female speech, woman speaking', 'Speech'],
    "5312870730.wav": ['Male speech, man speaking'],
    "4491342746.wav": ['Child speech, kid speaking', 'Children playing', 'Speech'],
    "5317259686.wav": ['Female speech, woman speaking', 'Laughter', 'Speech'],
    "4491371230.wav": ['Speech'],
    "5319163326.wav": ['Drum kit', 'Musical instrument'],
    "4508384185.wav": ['Baby laughter', 'Laughter'],
    "5323209509.wav": ['Child speech, kid speaking', 'Frying (food)', 'Inside, small room', 'Speech'],
    "5323579442.wav": ['Piano'],
    "4510506060.wav": ['Dishes, pots, and pans', 'Speech'],
    "5328004991.wav": ['Child speech, kid speaking', 'Speech'],
    "4514212431.wav": ['Child speech, kid speaking', 'Speech'],
    "5332054914.wav": ['Breathing', 'Speech'],
    "5343144349.wav": ['Child speech, kid speaking', 'Speech'],
    "4516647839.wav": ['Animal', 'Dog', 'Speech'],
    "4518113460.wav": [],
    "5344655615.wav": ['Electric piano', 'Keyboard (musical)', 'Music', 'Musical instrument', 'Piano'],
    "4520112789.wav": ['Environmental noise', 'Speech'],
    "5348154608.wav": ['Baby cry, infant cry', 'Child speech, kid speaking', 'Laughter', 'Speech'],
    "5349766264.wav": ['Drum', 'Keyboard (musical)', 'Music', 'Musical instrument', 'Piano'],
    "4523714689.wav": ['Child speech, kid speaking', 'Children playing', 'Laughter'],
    "5350306994.wav": ['Child speech, kid speaking', 'Speech'],
    "4524702622.wav": ['Child speech, kid speaking', 'Music', 'Speech'],
    "5351925965.wav": ['Laughter', 'Speech'],
    "5352412169.wav": ['Inside, public space', 'Music', 'Speech'],
    "4528442941.wav": ['Child speech, kid speaking', 'Speech'],
    "5353485791.wav": ['Female speech, woman speaking', 'Laughter', 'Speech'],
    "5354842767.wav": ['Baby laughter'],
    "4532587548.wav": ['Child speech, kid speaking', 'Speech'],
    "4533035526.wav": ['Hiccup', 'Inside, small room', 'Rub', 'Speech'],
    "5356919058.wav": ['Music', 'Musical instrument', 'Singing'],
    "5363401248.wav": ['Boat, Water vehicle', 'Speech', 'Wind', 'Wind noise (microphone)'],
    "4533118162.wav": ['Laughter'],
    "5368243767.wav": ['Conversation', 'Female speech, woman speaking', 'Speech'],
    "4534736385.wav": ['Inside, large room or hall', 'Inside, public space', 'Speech'],
    "5379354799.wav": ['Speech'],
    "4553174386.wav": ['Child speech, kid speaking', 'Children playing', 'Speech'],
    "5388595493.wav": ['Child speech, kid speaking', 'Inside, small room', 'Speech'],
    "4561969559.wav": ['Music', 'Musical instrument', 'Plucked string instrument', 'Strum'],
    "5398268379.wav": ['Child singing', 'Child speech, kid speaking', 'Clapping', 'Singing', 'Speech'],
    "4564918802.wav": ['Child speech, kid speaking', 'Speech'],
    "5398860552.wav": ['Babbling', 'Child speech, kid speaking', 'Speech'],
    "5409193177.wav": ['Emergency vehicle', 'Music', 'Musical instrument', 'Outside, urban or manmade', 'Speech', 'Vehicle'],
    "4565306431.wav": ['Siren', 'Speech', 'Vehicle'],
    "5427285434.wav": ['Child speech, kid speaking', 'Speech'],
    "4567287291.wav": ['Babbling', 'Speech'],
    "5435156153.wav": ['Silence', 'Speech'],
    "5435506054.wav": ['Conversation', 'Female speech, woman speaking'],
    "4570227004.wav": ['Child speech, kid speaking', 'Dishes, pots, and pans'],
    "5449359673.wav": ['Speech'],
    "4571054955.wav": ['Music', 'Musical instrument', 'Piano'],
    "5456274822.wav": ['Guitar', 'Music', 'Musical instrument', 'Speech'],
    "4573118187.wav": ['Child speech, kid speaking', 'Speech'],
    "5459816888.wav": ['Acoustic guitar', 'Guitar', 'Music', 'Musical instrument', 'Singing'],
    "4584426085.wav": ['Female speech, woman speaking', 'Speech'],
    "5460164242.wav": ['Child speech, kid speaking', 'Music', 'Speech'],
    "4598189055.wav": ['Child speech, kid speaking', 'Inside, small room', 'Slap, smack', 'Speech'],
    "5468020076.wav": ['Female speech, woman speaking', 'Speech'],
    "4598213889.wav": ['Inside, small room', 'Speech', 'Television'],
    "4599618003.wav": ['Baby cry, infant cry', 'Female singing', 'Music', 'Speech'],
    "5476140602.wav": ['Train'],
    "4603486590.wav": ['Guitar', 'Music', 'Musical instrument'],
    "5479373398.wav": ['Children playing', 'Speech'],
    "4604160981.wav": ['Child speech, kid speaking', 'Male speech, man speaking', 'Speech'],
    "5482316216.wav": [],
    "5489050159.wav": ['Speech'],
    "4609275864.wav": ['Children playing', 'Inside, public space'],
    "4615486172.wav": ['Inside, small room', 'Music', 'Singing', 'Speech'],
    "5494269389.wav": ['Male speech, man speaking', 'Rain', 'Rain on surface', 'Speech'],
    "5498241342.wav": ['Child speech, kid speaking', 'Children playing', 'Speech'],
    "4622219704.wav": ['Rain', 'Rain on surface', 'Speech'],
    "5498668540.wav": ['Baby laughter', 'Child speech, kid speaking', 'Music', 'Speech'],
    "4633647136.wav": ['Clapping', 'Hands', 'Inside, small room', 'Singing', 'Speech'],
    "4635309062.wav": ['Frying (food)', 'Rain', 'Speech'],
    "5502665217.wav": ['Electric guitar', 'Guitar', 'Music', 'Musical instrument', 'Plucked string instrument', 'Rock music'],
    "4636208292.wav": ['Cutlery, silverware', 'Dishes, pots, and pans', 'Speech'],
    "5503778096.wav": ['Child speech, kid speaking', 'Speech'],
    "5504106604.wav": ['Drum', 'Electric guitar', 'Guitar', 'Music', 'Musical instrument', 'Plucked string instrument', 'Strum'],
    "4637804560.wav": ['Babbling', 'Baby cry, infant cry', 'Female speech, woman speaking', 'Inside, small room', 'Speech'],
    "5504570766.wav": ['Meow', 'Speech', 'Vehicle'],
    "4647070508.wav": ['Child singing', 'Music'],
    "5511691883.wav": ['Wind', 'Wind noise (microphone)'],
    "4648608582.wav": ['Singing'],
    "5521781780.wav": ['Children playing', 'Outside, urban or manmade', 'Speech'],
    "4652947202.wav": ['Male speech, man speaking', 'Speech'],
    "5532742171.wav": ['Music', 'Speech'],
    "4653911781.wav": ['Baby laughter', 'Laughter', 'Speech'],
    "5540466660.wav": ['Child speech, kid speaking', 'Dishes, pots, and pans', 'Speech'],
    "4654125833.wav": ['Baby cry, infant cry', 'Music'],
    "5541230874.wav": ['Child speech, kid speaking', 'Music', 'Speech'],
    "4654222137.wav": ['Speech', 'Vehicle'],
    "5551527400.wav": ['Child speech, kid speaking'],
    "4657443720.wav": ['Domestic animals, pets', 'Musical instrument', 'Speech'],
    "4657452456.wav": ['Child speech, kid speaking', 'Domestic animals, pets', 'Music', 'Speech'],
    "5552271623.wav": ['Gunshot, gunfire', 'Speech', 'Vehicle'],
    "4658396458.wav": ['Babbling', 'Tick-tock'],
    "4660653323.wav": ['Speech', 'Walk, footsteps'],
    "5554974582.wav": ['Child speech, kid speaking', 'Speech'],
    "5557455580.wav": ['Child speech, kid speaking', 'Speech'],
    "4669303210.wav": ['Sewing machine', 'Typewriter'],
    "5557751246.wav": ['Tap', 'Walk, footsteps'],
    "4671452046.wav": ['Female speech, woman speaking', 'Male speech, man speaking', 'Speech'],
    "5558701375.wav": ['Domestic animals, pets', 'Environmental noise', 'Speech'],
    "4680621645.wav": ['Babbling', 'Child speech, kid speaking', 'Speech'],
    "5560177516.wav": ['Mechanisms'],
    "5563273949.wav": ['Speech'],
    "4687190278.wav": ['Child speech, kid speaking', 'Music', 'Speech'],
    "4688033528.wav": ['Cough', 'Inside, small room', 'Speech', 'Throat clearing'],
    "5566028125.wav": ['Child speech, kid speaking', 'Speech'],
    "4691227795.wav": ['Baby cry, infant cry', 'Speech'],
    "5569333947.wav": ['Baby laughter'],
    "4691952596.wav": ['Child speech, kid speaking', 'Speech'],
    "5569974351.wav": ['Child speech, kid speaking', 'Patter', 'Speech'],
    "4692611470.wav": ['Child speech, kid speaking', 'Laughter'],
    "5570574289.wav": ['Baby cry, infant cry', 'Guitar'],
    "4697577963.wav": ['Electric guitar', 'Musical instrument'],
    "4699001028.wav": ['Guitar', 'Music', 'Musical instrument', 'Singing'],
    "5570785480.wav": ['Music', 'Musical instrument', 'Speech'],
    "4707679105.wav": ['Babbling', 'Child speech, kid speaking'],
    "5571838736.wav": ['Clapping', 'Hands', 'Music', 'Musical instrument', 'Singing', 'Speech', 'Tap'],
    "4715441335.wav": ['Babbling', 'Child speech, kid speaking', 'Music', 'Sewing machine', 'Speech'],
    "5572234385.wav": ['Babbling', 'Child speech, kid speaking', 'Speech'],
    "4715646346.wav": ['Male speech, man speaking', 'Speech'],
    "5573667855.wav": ['Inside, large room or hall', 'Inside, public space', 'Speech'],
    "4734391811.wav": ['Baby cry, infant cry', 'Speech'],
    "4748191834.wav": ['Singing'],
    "5581572407.wav": ['Speech'],
    "4755578412.wav": ['Chatter', 'Dishes, pots, and pans', 'Inside, public space', 'Speech'],
    "5585306763.wav": ['Laughter', 'Speech'],
    "4756735316.wav": ['Child speech, kid speaking', 'Speech'],
    "5586292125.wav": ['Clapping', 'Speech'],
    "4759140711.wav": ['Hubbub, speech noise, speech babble', 'Inside, public space', 'Music', 'Speech'],
    "5586394125.wav": ['Outside, rural or natural', 'Speech', 'Vehicle', 'Wind', 'Wind noise (microphone)'],
    "4760692619.wav": ['Child singing', 'Child speech, kid speaking', 'Music', 'Speech', 'Television'],
    "5591479226.wav": ['Singing'],
    "4762946272.wav": ['Speech'],
    "5597037980.wav": ['Child singing', 'Child speech, kid speaking', 'Female speech, woman speaking', 'Music'],
    "4765717907.wav": ['Child speech, kid speaking', 'Speech'],
    "4769790180.wav": ['Laughter', 'Male speech, man speaking', 'Narration, monologue', 'Speech'],
    "5599111348.wav": ['Child speech, kid speaking', 'Clapping', 'Music', 'Musical instrument', 'Thump, thud'],
    "4772380747.wav": ['Music', 'Singing'],
    "5607360009.wav": ['Bird', 'Environmental noise'],
    "4772810451.wav": ['Animal', 'Dog', 'Domestic animals, pets'],
    "4773574474.wav": ['Children shouting', 'Crowd'],
    "5608080109.wav": ['Child speech, kid speaking', 'Clapping', 'Hands', 'Speech'],
    "4773661972.wav": ['Child speech, kid speaking', 'Speech'],
    "4782721238.wav": ['Speech'],
    "5608194207.wav": ['Cheering', 'Crowd'],
    "4783216249.wav": ['Conversation', 'Female speech, woman speaking'],
    "5614883965.wav": ['Baby laughter', 'Door', 'Speech'],
    "5625120077.wav": ['Ocean', 'Speech'],
    "4784365267.wav": ['Water tap, faucet'],
    "5630978721.wav": ['Baby cry, infant cry'],
    "5644588422.wav": ['Applause', 'Male speech, man speaking', 'Speech'],
    "4790249002.wav": ['Music', 'Musical instrument'],
    "5650441607.wav": ['Speech'],
    "4799461134.wav": ['Child singing', 'Child speech, kid speaking'],
    "5654616479.wav": ['Chuckle, chortle', 'Laughter', 'Speech'],
    "5656360308.wav": ['Hip hop music', 'Music', 'Rapping', 'Speech', 'Speech synthesizer'],
    "4800889464.wav": ['Child singing', 'Child speech, kid speaking', 'Speech'],
    "5662041273.wav": ['Child speech, kid speaking', 'Speech'],
    "4807584591.wav": ['Conversation', 'Female speech, woman speaking', 'Male speech, man speaking', 'Speech'],
    "4816724524.wav": ['Inside, public space', 'Speech'],
    "5662313254.wav": ['Air conditioning', 'Speech'],
    "5671008554.wav": ['Inside, large room or hall'],
    "4822859674.wav": ['Child speech, kid speaking', 'Speech'],
    "5681946487.wav": ['Bathtub (filling or washing)', 'Child speech, kid speaking', 'Speech', 'Splash, splatter', 'Water'],
    "4823372280.wav": ['Music'],
    "5687893796.wav": ['Music', 'Musical instrument'],
    "5694941758.wav": ['Female speech, woman speaking', 'Music', 'Wind chime'],
    "4833887719.wav": ['Child speech, kid speaking', 'Children playing', 'Inside, large room or hall', 'Speech'],
    "5697975018.wav": ['Guitar', 'Music', 'Musical instrument'],
    "5705234852.wav": ['Cheering', 'Crowd', 'Male speech, man speaking'],
    "4834742923.wav": ['Guitar', 'Music', 'Musical instrument', 'Singing'],
    "5710270308.wav": ['Child speech, kid speaking', 'Speech'],
    "4838398062.wav": ['Bird'],
    "5717610831.wav": ['Child singing', 'Child speech, kid speaking', 'Music', 'Singing'],
    "4848160422.wav": ['Railroad car, train wagon', 'Subway, metro, underground', 'Train'],
    "5722872813.wav": ['Keyboard (musical)'],
    "4849169953.wav": ['Child speech, kid speaking', 'Music'],
    "5728648619.wav": ['Female speech, woman speaking', 'Speech'],
    "4852243052.wav": ['Child speech, kid speaking', 'Outside, rural or natural', 'Snort', 'Speech', 'Walk, footsteps'],
    "4857781123.wav": ['Conversation', 'Speech'],
    "5729076991.wav": ['Babbling', 'Glockenspiel', 'Laughter', 'Music', 'Speech'],
    "4862215227.wav": ['Speech'],
    "5731927400.wav": ['Domestic animals, pets', 'Speech'],
    "4867392579.wav": ['Children playing'],
    "5732384406.wav": ['Drum', 'Drum kit', 'Music', 'Musical instrument'],
    "4870520876.wav": ['Child speech, kid speaking', 'Clapping', 'Speech'],
    "5738937644.wav": ['Air conditioning', 'Male speech, man speaking', 'Speech'],
    "4874075304.wav": ['Wind noise (microphone)'],
    "4874416538.wav": ['Child speech, kid speaking', 'Speech'],
    "4876265566.wav": ['Cupboard open or close', 'Speech'],
    "5750799064.wav": ['Child speech, kid speaking', 'Speech'],
    "4882414082.wav": ['Baby cry, infant cry'],
    "5757232720.wav": ['Child speech, kid speaking', 'Screaming'],
    "4882821564.wav": ['Child speech, kid speaking', 'Crumpling, crinkling', 'Speech'],
    "4885464111.wav": ['Baby cry, infant cry', 'Speech'],
    "5759653927.wav": ['Laughter', 'Male speech, man speaking', 'Music', 'Speech'],
    "4888273655.wav": [],
    "5765473409.wav": ['Inside, small room', 'Speech'],
    "4889681401.wav": ['Speech', 'Vehicle'],
    "5766018720.wav": ['Animal', 'Horse', 'Music', 'Musical instrument', 'Neigh, whinny', 'Printer', 'Speech', 'Synthetic singing'],
    "4891273513.wav": ['Conversation', 'Female speech, woman speaking', 'Laughter', 'Speech'],
    "5770462342.wav": ['Child speech, kid speaking', 'Speech'],
    "5770740059.wav": ['Guitar', 'Music', 'Musical instrument', 'Singing'],
    "4892615149.wav": ['Child singing', 'Child speech, kid speaking'],
    "4903182028.wav": ['Child speech, kid speaking', 'Ocean', 'Speech', 'Waves, surf', 'Wind'],
    "5779635831.wav": ['Animal', 'Run'],
    "4905439409.wav": ['Bird', 'Cough', 'Speech'],
    "4915733559.wav": ['Child singing', 'Dishes, pots, and pans', 'Speech'],
    "5787622236.wav": ['Child speech, kid speaking', 'Speech'],
    "5793991791.wav": ['Child speech, kid speaking', 'Children playing', 'Laughter', 'Speech'],
    "4916201843.wav": ['Babbling', 'Laughter', 'Speech'],
    "5798851653.wav": ['Children playing', 'Speech'],
    "4919726862.wav": ['Outside, rural or natural', 'Speech'],
    "5799956920.wav": ['Boat, Water vehicle', 'Speech', 'Splash, splatter', 'Water'],
    "4930421543.wav": ['Music', 'Singing'],
    "5802161982.wav": ['Inside, large room or hall'],
    "10001787725.wav": ['Humming', 'Speech'],
    "4938136806.wav": ['Motorboat, speedboat', 'Propeller, airscrew', 'Wind noise (microphone)'],
    "10011660054.wav": ['Female speech, woman speaking', 'Speech'],
    "4938937283.wav": ['Inside, small room', 'Speech'],
    "10041199716.wav": ['Child speech, kid speaking', 'Female speech, woman speaking'],
    "4939407002.wav": ['Child speech, kid speaking', 'Dishes, pots, and pans', 'Humming', 'Speech'],
    "10095741516.wav": ['Hip hop music', 'Music', 'Musical instrument', 'Speech'],
    "4949582792.wav": ['Baby cry, infant cry'],
    "10130549263.wav": ['Cutlery, silverware', 'Dishes, pots, and pans', 'Speech'],
    "4953208186.wav": ['Chuckle, chortle', 'Inside, small room', 'Laughter', 'Male speech, man speaking', 'Narration, monologue', 'Snicker', 'Speech'],
    "4963040001.wav": ['Babbling', 'Inside, small room', 'Music', 'Speech'],
}

sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
nlp_svo = spacy.load('en_core_web_trf')
embedding_cache = {}

# === Utility Functions ===
def get_embedding_with_cache(text):
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = sbert_model.encode([text], show_progress_bar=False)
        embedding_cache[text] = embedding
        return embedding

def compute_similarity(text1, text2):
    emb1 = get_embedding_with_cache(text1)
    emb2 = get_embedding_with_cache(text2)
    return float(util.cos_sim(emb1, emb2)[0][0])

def boost_confidence_ratio(audio_tags, caption_elements, audio_captions, alpha=0.5, caption_similarity_threshold=0.5):
    num_captions = len(audio_captions)
    results = {}
    for tag, audio_conf in audio_tags:
        match_count = sum(
            1 for word in caption_elements if compute_similarity(word, tag) > caption_similarity_threshold
        )
        match_ratio = min(1, match_count / num_captions) if num_captions > 0 else 0
        boosted_conf = min(1, alpha * match_ratio + (1 - alpha) * audio_conf)
        results[tag] = {
            'original': audio_conf,
            'boosted': boosted_conf,
            'matches': match_count,
            'match_ratio': f"{match_count}/{num_captions} ({match_ratio:.3f})",
        }
    return results

def extract_caption_phrases(captions_data):
    connected_phrases = {}
    for key, captions in captions_data.items():
        audio_captions = captions.get("audio_captions", [])
        unique_phrases = set()
        for caption in audio_captions:
            doc = nlp_svo(caption)
            used_token_idxs = set()
            
            # Step 1: SVO triplets
            for token in doc:
                if token.pos_ == "VERB":
                    subj = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                    obj = [child for child in token.children if child.dep_ in ("dobj", "attr", "prep", "pobj")]
                    if subj and obj:
                        all_tokens = subj + [token] + obj
                    elif subj:
                        all_tokens = subj + [token]
                    elif obj:
                        all_tokens = [token] + obj
                    else:
                        continue

                    token_idxs = {t.i for t in all_tokens}
                    if not token_idxs & used_token_idxs:
                        phrase = " ".join(
                            t.text.lower() for t in all_tokens
                            if not t.is_stop and not t.is_punct
                        ).strip()
                        if phrase:
                            unique_phrases.add(phrase)
                            used_token_idxs.update(token_idxs)

            # Step 2: Noun chunks
            for chunk in doc.noun_chunks:
                token_idxs = {t.i for t in chunk}
                if not token_idxs & used_token_idxs:
                    filtered = [t.text.lower() for t in chunk if not t.is_stop and not t.is_punct]
                    phrase = " ".join(filtered).strip()
                    if phrase:
                        unique_phrases.add(phrase)
                        used_token_idxs.update(token_idxs)

            # Step 3: Compound noun phrases
            for token in doc:
                if token.dep_ == "compound" and token.head.pos_ == "NOUN":
                    token_idxs = {token.i, token.head.i}
                    if not token_idxs & used_token_idxs:
                        if not token.is_stop and not token.is_punct and not token.head.is_stop and not token.head.is_punct:
                            phrase = f"{token.text.lower()} {token.head.text.lower()}".strip()
                            if phrase:
                                unique_phrases.add(phrase)
                                used_token_idxs.update(token_idxs)

        connected_phrases[key] = {"all_phrases": list(unique_phrases)}
    return connected_phrases




def is_semantic_match(predicted_tag, ground_truth_tags, threshold):
    pred_emb = get_embedding_with_cache(predicted_tag)
    gt_embs = sbert_model.encode(ground_truth_tags, show_progress_bar=False)
    if gt_embs.ndim == 1:
        gt_embs = gt_embs.unsqueeze(0)
    sims = util.cos_sim(pred_emb, gt_embs)[0]
    return sims.max().item() >= threshold if sims.numel() > 0 else False

def evaluate_combination(clipwise_tags, ground_truth_list, caption_elements, audio_captions,
                         alpha, confidence_threshold, caption_similarity_threshold):
    boosted_results = boost_confidence_ratio(clipwise_tags, caption_elements, audio_captions, alpha, caption_similarity_threshold)
    selected_tags = [tag for tag, info in boosted_results.items() if info['boosted'] >= confidence_threshold]

    true_positives = len([tag for tag in selected_tags if tag in ground_truth_list])
    false_positives = len([tag for tag in selected_tags if tag not in ground_truth_list])
    false_negatives = len([gt for gt in ground_truth_list if gt not in selected_tags])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1, selected_tags, ground_truth_list, boosted_results

# === Main Evaluation ===
if __name__ == "__main__":
    with open(json_file_path, 'r') as f:
        captions_data = json.load(f)

    connected_phrases = extract_caption_phrases(captions_data)
    alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    confidence_thresholds = [0.3,0.5]
    caption_similarity_thresholds = [0.3, 0.5]
    results_combinations = []

    for alpha in alpha_values:
        for conf_thresh in confidence_thresholds:
            for cap_sim_thresh in caption_similarity_thresholds:
                print(f"\nEvaluating: Alpha={alpha}, Conf={conf_thresh}, CaptionSim={cap_sim_thresh}")
                precision_total, recall_total, f1_total = 0.0, 0.0, 0.0
                clip_count = 0

                for raw_fname, captions in list(captions_data.items()):
                    fname = f"{raw_fname}.wav"
                    if fname not in predictions_df.index:
                        continue  # Skip files not in prediction CSV

                    caption_elements = connected_phrases.get(raw_fname, {}).get("all_phrases", [])
                    audio_captions = captions.get("audio_captions", [])
                    ground_truth = ground_truth_tags_dict.get(fname, [])

                    if not ground_truth:
                        continue

                    row = predictions_df.loc[fname]
                    clipwise_tags = []
                    for i in range(1, 11):  # tag1 to tag10
                        tag = row.get(f'tag{i}')
                        prob = row.get(f'tag{i}prob')
                        if isinstance(tag, str) and not pd.isna(prob): #and float(prob) > 0.5:
                            clipwise_tags.append((tag, float(prob)))
                    """
                    """

                    precision, recall, f1, selected_tags, gt_tags, boosted_results = evaluate_combination(
                    clipwise_tags, ground_truth, caption_elements, audio_captions,
                    alpha, conf_thresh, cap_sim_thresh)

                    """
                    if fname in ("5354842767.wav","5363401248.wav","5398268379.wav","5427285434.wav"):
                        print(f"filename: {fname}")
                        print(f"caption element: {caption_elements}")
                        print(f"audio_captions: {audio_captions}")
                        print(f"ground_truth: {ground_truth}")
                        print(f"boosted_results: {boosted_results}")
                        print("")
                    """

                    precision_total += precision
                    recall_total += recall
                    f1_total += f1
                    clip_count += 1

                if clip_count > 0:
                    results_combinations.append({
                        'alpha': alpha,
                        'conf_thresh': conf_thresh,
                        'caption_sim_thresh': cap_sim_thresh,
                        'precision': precision_total / clip_count,
                        'recall': recall_total / clip_count,
                        'f1': f1_total / clip_count,
                    })

    results_df = pd.DataFrame(results_combinations)
    print("\nSummary:")
    print(tabulate(results_df.round(3), headers='keys', tablefmt='grid', showindex=False))
    results_df.to_csv("evaluation_val_4.1.csv", index=False)

    plt.figure(figsize=(10, 6))
    for cap_sim_thresh in sorted(results_df['caption_sim_thresh'].unique()):
        for conf_thresh in sorted(results_df['conf_thresh'].unique()):
            subset = results_df[
                (results_df['caption_sim_thresh'] == cap_sim_thresh) &
                (results_df['conf_thresh'] == conf_thresh)
            ].sort_values(by='alpha')
            plt.plot(subset['alpha'], subset['f1'], marker='o',
                     label=fr'$\tau_{{sim}}$={cap_sim_thresh}, $\tau_{{pred}}$={conf_thresh}')
    plt.xlabel("α", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

