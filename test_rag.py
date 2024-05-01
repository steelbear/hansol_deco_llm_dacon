import csv
from string import Template

import chromadb
import torch

from chromadb.utils import embedding_functions
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

device = 'auto'
fine_tuned_model = 'SteelBear/open-solar-ko-hansol-16-8'
ef_model_name = 'distiluse-base-multilingual-cased-v1'

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    num_beams=2,
    repetition_penalty=1.2,
    length_penalty=0.5,
    )


def test_data_generator():
    with open('data/test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip the header in csvfile
        
        for id, text in reader:
            yield {'id': id, 'input': text}


def generate(model, generation_config, template, split_token, **inputs_dict):
    prompt = template.substitute(**inputs_dict)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(inputs, generation_config=generation_config)
    output_text = tokenizer.batch_decode(outputs)[0]
    output_text = output_text.split(split_token)[-1]
    output_text = output_text.replace("</s>", "")

    return output_text


def question_answer(user_input, collection):
    results = collection.query(
        query_texts=[user_input],
        n_results=2
    )

    return results['documents'][0]


def summarization(user_input, knowledge_list, generation_config):
    template = Template(
"""Answer the user's questions in as much detail as possible. Make sure to write your answers based on the sub-question given and the relevant knowledge.

${knowledges}

Question:
${user}
Answer:
"""
    )
    split_token = "Answer:\n"

    knowledges = '\n\n'.join(knowledge_list)

    output = generate(model, generation_config, template, split_token, user=user_input, knowledges=knowledges)

    return output


if __name__ == '__main__':
    # load the inference model, embedding model and tokenizer
    #torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        fine_tuned_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
    embedding_model = SentenceTransformer(ef_model_name)

    client = chromadb.PersistentClient(path='./vectordb')
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=ef_model_name)
    collection = client.get_collection(name='hansol', embedding_function=sentence_transformer_ef)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    if pad == None or pad == eos:
        tokenizer.pad_token_id = 0
    print("length of tokenizer:", len(tokenizer))

    generation_config.bos_token_id = tokenizer.bos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.pad_token_id = tokenizer.pad_token_id

    # load test dataset and preprocess
    test_dataset = Dataset.from_generator(test_data_generator)
    test_dataset.set_format(type="torch")

    # generate answers in test dataset and embed by a document
    with open('inference.csv', 'w', newline='\n') as infer_f:
        infer_writer = csv.writer(infer_f)
        infer_writer.writerow(['id', '질문', '답변'])
        with open('submission.csv', 'w', newline='\n') as submit_f:
            submit_writer = csv.writer(submit_f)
            submit_writer.writerow(['id'] + ['vec_%d' % i for i in range(512)])
    
            for example in tqdm(test_dataset):
                knowledges = question_answer(example['input'], collection)
                output = summarization(example['input'], knowledges, generation_config)

                infer_writer.writerow([example['id'], example['input'], output])
                vector = embedding_model.encode(output)
                submit_writer.writerow([example['id']] + vector.tolist())
