import csv
from string import Template

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm

device = 'auto'
fine_tuned_model = 'SteelBear/llama-2-koen-hansol'

gk_prompt_template = Template(
    """${system}

${examples}

Input: ${user}
Knowledge: 
1) """)

gk_system_input = "Generate some interior knowledge about the input. For example,"
gk_examples = """Input: 철골구조의 단점이 뭐야?
Knowledge: 
1) 철골구조는 화재 발생시 철골이 녹아 건물의 붕괴로 이어질 수 있다.
2) 철골 자체가 비교적 비싼 재료다.
3) 철골구조는 부식에 따른 유지보수가 필요하다.

Input: 철골철근 콘크리트 구조가 뭐야?
Knowledge: 
1) 철골철근 콘크리트 구조는 강철 골조와 강철 철근, 그리고 콘크리트를 함께 사용하여 건축물을 구성하는 구조체다.
2) 철골철근 콘크리트 구조는 철골의 강도와 내구성, 철근의 인장력, 그리고 콘크리트의 압축력을 조합하여 건축물에 대한 강성과 내구성을 향상시킨다.
3) 철골철근콘크리트는 철골과 철근콘크리트의 장점을 결합하여 안정성을 높이고 건물을 지탱하는 구조다."""

prompt_template = Template(
    """${system}

${examples}

Input: ${user}
Knowledge: ${knowledge}
Output: """
)

system_input = """Using the given knowledge, create a response to the input following these options:
- Your answer have more than 5 sentences.
- You must write the answer in Korean.
- Your first sentence should briefly introduce what you're about to describe.
- If multiple concepts appear, you should explain them one by one.
- Your answer should be detailed enough to be easy to understand.
- Summarize the entirety of your answer with a final sentence.

For example:"""

examples = """Input: 면진 구조와 제진 구조의 차이점이 뭐야?
Knowledge: 면진구조는 지진력이 구조물에 상대적으로 약하게 전달되도록 수평방향으로 유연한 장치를 이용하여 지면과 건물을 분리시킴으로 건물에 전달되는 지진의 영향을 최소화하는 구조다. 제진구조는 지진발생 시 구조물로 전달된 지진력을 건물 내외부에 특수한 장치(탬퍼 등)을 설치하여 지진에 대한 구조물의 응답을 제어하는 구조를 말한다.
Output: 면진구조와 제진구조는 지진으로 인한 진동을 줄이는 데 사용되는 두 가지 다른 방법입니다. 면진구조는 건물과 지반 사이에 일정한 간격을 두어 지진으로 인한 진동을 완화시키는 방법입니다. 이 간격은 지진 발생 시 건물과 지반 사이의 진동을 흡수하여 건물에 전달되는 충격을 줄입니다. 반면에 제진구조는 건물 내부에서 진동을 제어하는 방법입니다. 제진장치는 건물 내부에 설치되어 지진으로 인한 진동을 감소시킵니다. 제진장치는 일반적으로 탬퍼처럼 내외부에 설치된 특수한 장치를 사용하여 건물의 진동을 제어하고, 에너지 소산을 통해 진동을 줄입니다. 요약하자면, 면진구조는 건물과 지반 사이의 간격을 두어 진동을 완화시키는 방법이고, 제진구조는 건물 내부에서 진동을 제어하는 방법입니다."""

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


def generated_knowledge(user_input, generation_config):
    gk_prompt = gk_prompt_template.substitute(system=gk_system_input, examples=gk_examples, user=user_input)
    inputs = tokenizer.encode(gk_prompt, return_tensors="pt").to('cuda')

    outputs = model.generate(inputs, generation_config=generation_config)
    output_text = tokenizer.batch_decode(outputs)[0]
    #print("### Generated Knowledge ###")
    #print(output_text)
    #print("### Generated Knowledge ###\n\n")

    output_text = output_text.split("Knowledge: \n")[3]
    #output_text = output_text.split("\n")[0]
    output_text = output_text.replace("</s>", "")
    return output_text


def question_answer(user_input, gk_generation_config, generation_config):
    gk_result = generated_knowledge(user_input, gk_generation_config)

    prompt = prompt_template.substitute(system=system_input, examples=examples, user=user_input, knowledge=gk_result)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

    outputs = model.generate(inputs, generation_config=generation_config)
    output_text = tokenizer.batch_decode(outputs)[0]
    #print("### Question Answering ###")
    #print(output_text)
    #print("### Question Answering ###\n\n")

    output_text = output_text.split("Output: ")[2]
    output_text = output_text.replace("</s>", "")

    return output_text


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
        infer_writer.writerow(['id', '질문', 'Generated Knowledge', '답변'])
        with open('submission.csv', 'w', newline='\n') as submit_f:
            submit_writer = csv.writer(submit_f)
            submit_writer.writerow(['id'] + ['vec_%d' % i for i in range(512)])
    
            for example in tqdm(test_dataset):
                output, gk_result = question_answer(example['input'], gk_generation_config, generation_config)
                infer_writer.writerow([example['id'], example['input'], gk_result, output])
                vector = embedding_model.encode(output)
                submit_writer.writerow([example['id']] + vector.tolist())
