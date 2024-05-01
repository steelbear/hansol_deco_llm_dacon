import csv
from string import Template

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm


fine_tuned_model = './custom_LLM_final'
device = 'auto'

# SOLAR prompt template
prompt_template = Template(
    """### System:
${system}

### User:
${user}

### Assistant:
""")

generation_config = GenerationConfig(
    max_new_tokens=512,
    )


def test_data_generator():
    with open('data/test.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip the header in csvfile
        
        for id, text in reader:
            yield {'id': id, 'input': text}


def tokenize(example):
    prompt = prompt_template.substitute(
        system="당신은 건물 도배 전문가로써 User의 질문에 알맞은 답변을 제시하세요.",
        user=example['input'])
    tokenized = tokenizer.encode(prompt, return_tensors='pt')
    example['input_ids'] = tokenized
    return example


if __name__ == '__main__':
    # load the inference model, embedding model and tokenizer
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        fine_tuned_model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
    embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

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
    test_dataset = test_dataset.map(tokenize)

    # generate answers in test dataset and embed by a document
    with open('inference.csv', 'w', newline='\n') as infer_f:
        infer_writer = csv.writer(infer_f)
        infer_writer.writerow(['id', '질문', '답변'])
        with open('submission.csv', 'w', newline='\n') as submit_f:
            submit_writer = csv.writer(submit_f)
            submit_writer.writerow(['id'] + ['vec_%d' % i for i in range(512)])
    
            for example in tqdm(test_dataset):
                outputs = model.generate(
                    example['input_ids'].to('cuda'),
                    do_sample=True,
                    generation_config=generation_config,
                )
                output_text = tokenizer.batch_decode(outputs)[0]
                output_text = output_text.split("### Assistant:\n")[1]
                output_text = output_text.replace("</s>", "")
                infer_writer.writerow([example['id'], example['input'], output_text])
                vector = embedding_model.encode(output_text)
                submit_writer.writerow([example['id']] + vector.tolist())
