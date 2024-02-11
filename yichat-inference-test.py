from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
from threading import Thread, currentThread

# model_name = "01-ai/Yi-6B-Chat"
model_name = "01-ai/Yi-6B-Chat-4bits"

#run_mode = "cpu"
run_mode = "gpu"

top_k = 3
top_p = 0.95
temperature = 1
Is_debug = 0

def load_model(model_name,run_mode):
    temp = "cpu" if run_mode == "cpu" else "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=temp, torch_dtype="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    ## tokenizer.pad_token = tokenizer.eos_token
    return model,tokenizer

model,tokenizer = load_model(model_name,run_mode)

dataset = [
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'This is Kevin, how are you?'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'This is Eric, hello?'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'How do you do?'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'This is RX, hi?'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hi there, this is Mike!'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'This is Jaskon, how are you?'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'This is Robert, hello?'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'This is Tom, how do you do?'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'This is Jack, hi?'} ],
        [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'Hi there, this is Robert!'} ],
    ]

def show_memory(description=""):
    temp1gb = 1024.0 ** 3
    gpu_number = torch.cuda.device_count()
    #gpu_number = 1
    for i in range(gpu_number):
        t = torch.cuda.get_device_properties(i).total_memory / temp1gb
        #r0 = torch.cuda.max_memory_reserved(i) / temp1gb
        r = torch.cuda.memory_reserved(i) / temp1gb
        #a0 = torch.cuda.max_memory_allocated(i) / temp1gb
        a = torch.cuda.memory_allocated(i) / temp1gb
        f = (r - a)
        #f0 = (r0 - a0)
        tempString = "GPU VRAM: reserved {:.3f} GB, allocated {:.3f} GB, free {:.3f} GB, total {:.3f} GB".format( r, a, f, t)
        print(tempString + " ---> " + description)

def my_print(tlist,temp):
    if Is_debug == 0:
        return
    print("----------------------------------> " + temp)
    for x in range(len(tlist)):
        print("--------------> No {}".format(x))
        print(tlist[x])

def my_print_ex(tlist,temp):
    if Is_debug == 0:
        return
    print("----------------------------------> " + temp)
    for x in range(len(tlist)):
        print("--------------> No {}".format(x))
        for y in (tlist[x]):
            print(y)


'''
处理前：1个用户的历史会话信息。
[{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': 'This is Richard, how are you?'}]

apply_chat_template的具体功能：去掉标点符号、'Role'及'Content'等

处理后：增加了特殊字符<|im_start|>和<|im_end|>，还有assistant。
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
This is Richard, how are you?<|im_end|>
<|im_start|>assistant
'''


############################################################ Batch 批处理
def batch_inference(batch_size,generate_max_len,description,postion=0):  # postion，用于多线程，每个线程处理1个

    history = [] # 用来保存这N个用户的对话记录，原始提示及生成文本

    for x in range(batch_size):    # 从dataset中复制数据, 地址拷贝； history指向了dataset中的内容，修改History、会导致dataset变化
        history.append(dataset[x+postion]) # 多个用户的数据，2维List；每个用户采用1个List；List内部，包含历史对话，{‘system’ } {‘user’} {‘assistant’} {‘user’} ....
    my_print_ex(history,"the original prompts")

    input_strings = []
    for x in history:
        temp = tokenizer.apply_chat_template(conversation=x, tokenize=False, add_generation_prompt=True)
        # print(x,temp)
        input_strings.append(temp)           # 只应用chat模版，添加起始及终止字符，添加初始assistat，不转换为Token ID
    my_print(input_strings,"apply the chat template on the original prompts")

    my_print([], "start the generation")
    inputs = tokenizer(input_strings, padding=True, return_tensors='pt')  # 不一样长，所以要加pad,在左边加pad；返回PyTorch Tensor，多维的
    if Is_debug != 0:
        print("Input Number: {}, Length of Input {}:".format(inputs.input_ids.shape[0], inputs.input_ids.shape[1]))
        print(inputs.input_ids)

    if (run_mode != "cpu"):
        inputs.to('cuda')

    real_length = generate_max_len + inputs.input_ids.shape[1]  # 已经padding，等长

    start_time = time.time()

    outputs = model.generate(
        inputs.input_ids, max_length=real_length,
        do_sample=True, repetition_penalty=1.3, no_repeat_ngram_size=5,
        temperature=temperature, top_k=top_k, top_p=top_p)

    end_time = time.time()
    print("Generated {:2n} x {:3n} tokens and took {:.3f} s".format(batch_size, generate_max_len, end_time - start_time),end="; ")
    show_memory(description)

    output_strings = tokenizer.batch_decode(outputs)  # 返回Tensor， Number x Length, 返回包含特殊字符，<|im_start|>和<|im_end|>等等
    my_print(output_strings, "the raw generated texts")

    '''
    for x in range(batch_size):
        temp = outputs[x][inputs.input_ids.shape[1]:]  # 在输出ID中，去掉输入ID；temp引用了output中的部分内容
        del outputs[x] # 这个地方可能存在内存泄漏
        output_string = tokenizer.decode(temp, skip_special_tokens=True)
        # print(output_string)

        history[x].append({'role': 'assistant', 'content': output_string})

    my_print_ex(history, "the final results")
    '''

'''
在输出中去掉输入，有2中实现方案：

1）在Token ID中实现（已经添加了pad的），直接在输出ID中去掉输入ID，小心VRAM泄露。

2）在Token中实现，存在一下问题：

输入为：<|im_start|>system，不含“ ”；并且没有做pad。
--------------> No 1
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
This is Eric, hello?<|im_end|>
<|im_start|>assistant

输出为：<unk><unk><|im_start|> system，多处了“ ”和pad。
--------------> No 1
<unk><unk><|im_start|> system
You are a helpful assistant.<|im_end|> 
<|im_start|> user
This is Eric, hello?<|im_end|> 
<|im_start|> assistant
Hello! I am Yi, an AI assistant developed by 01.

'''


############################################################ Python Multithread 多线程
def multithread_test(batch_size,generate_max_len,description):
    start_time = time.time()

    t = []
    for x in range(batch_size):
        temp = Thread(target=batch_inference, args=(1,generate_max_len,description+", Child "+str(x),x))
        temp.start()
        t.append(temp)
    for x in range(batch_size):
        t[x].join()

    end_time = time.time()
    print("Generated {:2n} x {:3n} tokens and took {:.3f} s".format(batch_size,generate_max_len,end_time-start_time), end="; ")
    show_memory(description)


if __name__ == "__main__":

    batch_inference(2, 16, 'Warm up')

    batch_inference(1, 128, 'Batch')
    batch_inference(2, 128, 'Batch')
    batch_inference(3, 128, 'Batch')
    batch_inference(4, 128, 'Batch')
    batch_inference(5, 128, 'Batch')


    multithread_test(1, 128, 'Multithread')
    multithread_test(2, 128, 'Multithread')
    #multithread_test(3, 128, 'Multithread')
    #multithread_test(4, 128, 'Multithread')
    #multithread_test(5, 128, 'Multithread')





