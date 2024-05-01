import csv
from string import Template

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

device = 'auto'
fine_tuned_model = './custom_LLM_final'

gk_generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    repetition_penalty=1.1,
    top_k=3,
    temperature=0.95
    )

generation_config = GenerationConfig(
    max_new_tokens=512,
    do_sample=True,
    repetition_penalty=1.2,
    temperature=0.98,
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


def analysis_question(user_input, generation_config):
    template = Template(
"""In User's input, list what the User's questions are in the form of a list in Korean. For example,
Input: 
방청 페인트의 종류에는 어떤 것들이 있는지 알고 계신가요? 또한, 원목사이딩을 사용하는 것에 어떤 단점이 있을까요?
Output:
1. 방청 페인트의 종류에는 어떤 것들이 있는지 알고 계신가요?
2. 원목사이딩을 사용하는 것에 어떤 단점이 있을까요?

Input:
철골구조를 사용하는 고층 건물에서, 단열 효과를 높이기 위한 시공 방법은 무엇이 있을까요?
Output:
1. 철골구조를 사용하는 고층 건물에서, 단열 효과를 높이기 위한 시공 방법은 무엇이 있을까요?

Input:
{user}
Output:
1. """)
    split_token = "Output:\n"

    output = generate(model, generation_config, template, split_token, user=user_input)
    question_list = output.split('\n')

    return question_list


def question_answer(user_input, generation_config):
    template = Template(
"""Generate some interior knowledge about the input. For example,

Input: 
철골구조의 단점이 뭐야?
Knowledge: 
1. 철골구조는 화재 발생시 철골이 녹아 건물의 붕괴로 이어질 수 있다.
2. 철골 자체가 비교적 비싼 재료다.
3. 철골구조는 부식에 따른 유지보수가 필요하다.

Input: 
철골철근 콘크리트 구조가 뭐야?
Knowledge: 
1. 철골철근 콘크리트 구조는 강철 골조와 강철 철근, 그리고 콘크리트를 함께 사용하여 건축물을 구성하는 구조체다.

Input:
${user}
Knowledge:
1. """
    )
    split_token = "Knowledge:\n"
    
    knowledges = generate(model, generation_config, template, split_token, user=user_input)

    return knowledges


def summarization(user_input, question_list, knowledge_list, generation_config):
    template = Template(
"""Answer the user's questions in as much detail as possible. Make sure to write your answers based on the sub-question given and the relevant knowledge.

${knowledges}

Question:
${user}
Answer:
"""
    )
    split_token = "Answer:\n"

    knowledges = '\n\n'.join([
        f"Sub-question:\n{subquestion}\nKnowledge:\n{knowledge}"
        for subquestion, knowledge in zip(question_list, knowledge_list)
    ])

    output = generate(model, generation_config, template, split_token, user=user_input, knowledge_list=knowledges)

    return output


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

    # generate answers in test dataset and embed by a document
    with open('inference.csv', 'w', newline='\n') as infer_f:
        infer_writer = csv.writer(infer_f)
        infer_writer.writerow(['id', '질문', '답변'])
        with open('submission.csv', 'w', newline='\n') as submit_f:
            submit_writer = csv.writer(submit_f)
            submit_writer.writerow(['id'] + ['vec_%d' % i for i in range(512)])
    
            for example in tqdm(test_dataset):
                subquestions = analysis_question(example['input'], generation_config)
                knowledges = []
                for subquestion in subquestions:
                    knowledges.append(question_answer(subquestion, gk_generation_config))
                output = summarization(example['input'], subquestions, knowledges, generation_config)

                infer_writer.writerow([example['id'], example['input'], output])
                vector = embedding_model.encode(output)
                submit_writer.writerow([example['id']] + vector.tolist())
