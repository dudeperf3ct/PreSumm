import json
import pprint
import collections
import time
import numpy as np
import spacy
import os
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

# ## for multiprocessing
# from fastprogress.fastprogress import master_bar, progress_bar
# import concurrent
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# def num_cpus()->int:
#     "Get number of cpus"
#     try:                   return len(os.sched_getaffinity(0))
#     except AttributeError: return os.cpu_count()

# _default_cpus = min(16, num_cpus())

# def ifnone(a,b):
#     "`a` if `a` is not None, otherwise `b`."
#     return b if a is None else a

# def parallel(func, arr, max_workers:int=None, leave=False):
#     "Call `func` on every element of `arr` in parallel using `max_workers`."
#     max_workers = ifnone(max_workers, _default_cpus)
#     if max_workers<2: results = [func(o,i) for i,o in progress_bar(enumerate(arr), total=len(arr), leave=leave)]
#     else:
#         with ProcessPoolExecutor(max_workers=max_workers) as ex:
#             futures = [ex.submit(func,o,i) for i,o in enumerate(arr)]
#             results = []
#             for f in progress_bar(concurrent.futures.as_completed(futures), total=len(arr), leave=leave): 
#                 results.append(f.result())
#     if any([o is not None for o in results]): return results

input_json_file = "data/news-05022020.json"
output_json_file = "data/new_news-05022020.json"
news_file = "data/news.txt"
abs_summary_file = "data/news_abs.-1.candidate"
ext_summary_file = "data/news_ext_step-1.candidate"

with open(input_json_file) as f:
    data = [json.loads(line) for line in f]

new_data = defaultdict()
new_data = deepcopy(data)

# how many entries of data to process --> ideally, len(data)
num = 15000

# print first two records 
pprint.pprint(data[:2])

"""
data                    ---> Json contains Previous data
new_data                ---> Json Contains Previous data + 
new_ne_title            ---> add
new_ne_title_str        ---> add
new_abs_summary         ---> add (abstractive)
new_ne_abs_summary_str  ---> add (abstractive)
new_ext_summary         ---> add (extractive)
new_ne_ext_summary_str  ---> add (extractive)
"""

news = []
for i in range(num):
    news.append(data[i]['text'])

titles = []
for i in range(num):
    titles.append(data[i]['title'])
    
with open(news_file, "w") as f:
  for n in news:
    f.write(f'{n}\n')

# #########################################
# #   Transformer Abstractive Summarizer  #    
# #########################################
# # change ne_summary and ne_summary_str
# # change visible_gpu to gpu_ids

## First run: For the first time, you should use single-GPU, so the code can download the BERT model. 
## Use -visible_gpus -1, after downloading, you could kill the process and rerun the code with multi-GPUs.

start = time.time()
os.chdir("PreSumm/src")
os.system("python train.py -task abs -mode test_text -batch_size 3000 -text_src ../../data/news.txt -test_from ../baseline/cnndm_baseline_best.pt -log_file ../logs/test_abs_bert_cnndm -model_path MODEL_PATH -sep_optim true -use_interval true -visible_gpus 0,1 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 50 -result_path ../../data/news_abs")
os.chdir("../../")

print ("Time taken for abstractive summary :", time.time()-start)

#######################################
#   Transformer Extractive Summarizer #    
#######################################
# change ne_summary and ne_summary_str

start = time.time()
os.chdir("PreSumm/src")
os.system("python train.py -task ext -mode test_text -text_src ../../data/news.txt -test_from ../baseline/bertext_cnndm_transformer.pt -model_path MODEL_PATH -visible_gpus 1 -batch_size 3000  -log_file ../logs/test_ext_bert_cnndm -use_interval true -max_pos 512 -max_length 2000 -alpha 0.95 -min_length 50 -result_path ../../data/news_ext")
os.chdir("../../")

print ("Time taken for extractive summary :", time.time()-start)


abs_summ = []
with open(abs_summary_file, "r") as f:
    abs_summ = [line.rstrip() for line in f]

ext_summ = []
with open(ext_summary_file, "r") as f:
    ext_summ = [line.rstrip() for line in f]

# ###############################
# #   Spacy - NER for title     #    
# ###############################
# # change ne_title and ne_title_str

start = time.time()
# load the NER model
nlp = spacy.load('en_core_web_lg')

def ner_title(i, value):
    # make a sentence
    d = nlp(titles[i])
    ner_title = collections.defaultdict()
    text = []
    for v in d.ents:
        text.append(v.text)
        ner_title[v.text] = {"name": v.text, "type" : v.label_}
    new_data[i].update({"new_ne_title" : dict(ner_title)})
    new_data[i].update({"new_ne_title_str" : (" ").join(text)})

# serial processing
for i in tqdm(range(num)):
    ner_title(i, 0)

# # parallel processing
# # instead of 1000 original number = num
# ind = [i for i in range(10)]
# parallel(ner_title, ind)

print ("-"*15)
print ("Time taken for NER of title:", time.time()-start)

# print first two records 
for i in range(10):
    print("Old:", data[i]['ne_title'])
    print("New:", new_data[i]['new_ne_title'])
    print("Old:", data[i]['ne_title_str'])
    print("New:", new_data[i]['new_ne_title_str'])

# ###############################
# #   Spacy - NER for summary   #    
# ###############################
# change summary, ne_summary and ne_summary_str
start = time.time()

def abs_ner_summary(i, value):
    # make a sentence
    d = nlp(abs_summ[i])
    ner_summ = collections.defaultdict()
    text = []
    for v in d.ents:
        text.append(v.text)
        ner_summ[v.text] = {"name": v.text, "type" : v.label_}
    new_data[i].update({"new_abs_summary" : [abs_summ[i].split("<q>")]})
    new_data[i].update({"new_ne_abs_summary" : dict(ner_summ)})
    new_data[i].update({"new_ne_abs_summary_str" : (" ").join(text)})

# serial processing
for i in tqdm(range(num)):
    abs_ner_summary(i, 0)

# # parallel processing
# # instead of 1000 original number = num
# ind = [i for i in range(10)]
# parallel(ner_summary, ind)

print ("-"*15)
print ("Time taken for NER of abstractive summary:", time.time()-start)

start = time.time()

def ext_ner_summary(i, value):
    # make a sentence
    d = nlp(ext_summ[i])
    ner_summ = collections.defaultdict()
    text = []
    for v in d.ents:
        text.append(v.text)
        ner_summ[v.text] = {"name": v.text, "type" : v.label_}
    new_data[i].update({"new_ext_summary" : [ext_summ[i].split(".")]})
    new_data[i].update({"new_ne_ext_summary" : dict(ner_summ)})
    new_data[i].update({"new_ne_ext_summary_str" : (" ").join(text)})

# serial processing
for i in tqdm(range(num)):
    ext_ner_summary(i, 0)

# # parallel processing
# # instead of 1000 original number = num
# ind = [i for i in range(10)]
# parallel(ner_summary, ind)

print ("-"*15)
print ("Time taken for NER of extractive summary:", time.time()-start)

# print first two records
for i in range(10):
    print("Old:", data[i]['summary'])
    print("Abstractive:", new_data[i]['new_abs_summary'])
    print("Extractive:", new_data[i]['new_ext_summary'])
    print("Abstractive:", new_data[i]['new_ne_abs_summary'])
    print("Extractive:", new_data[i]['new_ne_ext_summary'])
    print("Abstractive:", new_data[i]['new_ne_abs_summary_str'])
    print("Extractive:", new_data[i]['new_ne_ext_summary_str'])

# # #########################################
# # #  Save the new dataset to json file  # #    
# # #########################################

if not os.path.exists(output_json_file):
    open(output_json_file, 'w').write("[%s]" % "\n".join(map(json.dumps, new_data)))
