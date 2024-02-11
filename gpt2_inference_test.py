from transformers import GPT2LMHeadModel,GPT2Tokenizer
import torch
import time
from threading import Thread
from multiprocessing import Process, Queue, set_start_method

import asyncio


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


def load_model(model_name):
    model = GPT2LMHeadModel.from_pretrained(model_name,device_map="cuda:0", torch_dtype="auto") # 模型转入GPU
    model.eval() # 进入推理模式

    tokenizer = GPT2Tokenizer.from_pretrained(model_name,padding_side='left') # 从左面开始Padding
    tokenizer.pad_token = tokenizer.eos_token # 该模型没有提供pad, 这里用eos_token作为pad_token
    return model,tokenizer


############################################################ Batch 批处理
def batch_test(base_model,base_tokenizer,dataset,num,length,description,Is_Print):
    start_time = time.time()

    inputs = base_tokenizer(dataset[:num], return_tensors='pt', padding=True) # 输入是List，要求返回PyTorch Tensor，需要Padding、从左开始
    inputs.to('cuda') # 二维，[[],[],... ,[]]
    generated_text_samples = base_model.generate( inputs.input_ids, max_length=length, pad_token_id=base_tokenizer.eos_token_id )
    if(Is_Print):
        for i, beam in enumerate(generated_text_samples):
            print(f"{i} -----> {base_tokenizer.decode(beam, skip_special_tokens=True)}")

    end_time = time.time()

    print("Generated {:2n} x {:3n} tokens and took {:.3f} s".format(num,length,end_time-start_time), end="; ")
    show_memory(description)


############################################################ Python asyncio 协程
async def coroutine_single_infer(base_model,base_tokenizer,text,length,Is_Print):
    inputs = base_tokenizer(text, return_tensors='pt', padding=True)
    inputs.to('cuda') # 二维，[[],[],... ,[]]
#   await asyncio.sleep(0.1) #
    generated_text_samples = base_model.generate( inputs.input_ids, max_length=length, pad_token_id=base_tokenizer.eos_token_id )
    if(Is_Print):
        for i, beam in enumerate(generated_text_samples):
            print(f"{i} -----> {base_tokenizer.decode(beam, skip_special_tokens=True)}")

async def coroutine_test(base_model,base_tokenizer,dataset,num,length,description,Is_Print):
    start_time = time.time()

    tasks = [ coroutine_single_infer(base_model,base_tokenizer,dataset[x],length,Is_Print) for x in range(num) ] # 定义多个协程
    await asyncio.gather( * tasks ) # 启动协程，等待

    end_time = time.time()

    print("Generated {:2n} x {:3n} tokens and took {:.3f} s".format(num,length,end_time-start_time), end="; ")
    show_memory(description)


############################################################ Python Multithread 多线程
def single_infer(base_model,base_tokenizer,text,length,Is_Print):
    inputs = base_tokenizer(text, return_tensors='pt', padding=True)
    inputs.to('cuda') # 二维，[[],[],... ,[]]
    generated_text_samples = base_model.generate( inputs.input_ids, max_length=length, pad_token_id=base_tokenizer.eos_token_id )
    if(Is_Print):
        for i, beam in enumerate(generated_text_samples):
            print(f"{i} -----> {base_tokenizer.decode(beam, skip_special_tokens=True)}")

def multithread_test(base_model,base_tokenizer,dataset,num,length,description,Is_Print):
    start_time = time.time()

    t = []
    for x in range(num):
        temp = Thread(target=single_infer, args=(base_model,base_tokenizer,dataset[x],length,Is_Print))
        temp.start()
        t.append(temp)
    for x in range(num):
        t[x].join()

    end_time = time.time()

    print("Generated {:2n} x {:3n} tokens and took {:.3f} s".format(num,length,end_time-start_time), end="; ")
    show_memory(description)


############################################################ Python Multiprocess 多进程
def single_process_infer(model_name,iq ,oq ,jb,length,Is_Print):
    show_memory("Begin, "+jb)
    base_model = GPT2LMHeadModel.from_pretrained(model_name, device_map="cuda:0", torch_dtype="auto")  # 模型转入GPU
    base_model.eval()  # 进入推理模式
    base_tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')  # 从左面开始Padding
    base_tokenizer.pad_token = base_tokenizer.eos_token
    show_memory("The model is loaded, "+jb)

    oq.put("done") # 完成模型导入

    text = iq.get() ######################## Read the prompt

    start_time = time.time()

    inputs = base_tokenizer(text, return_tensors='pt', padding=True)
    inputs.to('cuda') # 二维，[[],[],... ,[]]
    generated_text_samples = base_model.generate( inputs.input_ids, max_length=length, pad_token_id=base_tokenizer.eos_token_id )
    if(Is_Print):
        for i, beam in enumerate(generated_text_samples):
            print(f"{i} -----> {base_tokenizer.decode(beam, skip_special_tokens=True)}")

    end_time = time.time()

    print("Generated {:2n} x {:3n} tokens and took {:.3f} s".format(1,length,end_time-start_time), end="; ")
    show_memory(jb)

    #temp = base_tokenizer.decode(generated_text_samples[0], skip_special_tokens=True)
    #oq.put(temp)
    oq.put(jb+" done") ######################## Returns the text and ends

def multiprocess_test(model_name,dataset,num,length,description,Is_Print):
    inputQueues = [] # from parent to child process
    outputQueues = []  # from child process to parent
    p = []

    for x in range(num):
        iq = Queue()
        oq = Queue()
        inputQueues.append(iq)
        outputQueues.append(oq)

        temp = Process(target=single_process_infer, args=(model_name,iq,oq, "Child Process "+str(x),length,Is_Print ) )
        temp.start()
        p.append(temp)

    print("Waiting until all processes have loaded the model..")
    for x in range(num):
       _ = outputQueues[x].get() # 确保 Child process 已经就绪（装入模型）

    start_time = time.time()

    for x in range(num):
        inputQueues[x].put( dataset[x] )  # Parent 给 Child 分发 prompt

    for x in range(num):
        result = outputQueues[x].get()  # Parent 从个 Child 读取 generated text
        #print(result)

    end_time = time.time()

    for x in range(num):
        p[x].join()

    print("Generated {:2n} x {:3n} tokens and took {:.3f} s".format(num,length,end_time-start_time), end="; ")
    #show_memory(description)


############################################################ Main

if __name__ == "__main__":


    set_start_method('spawn')
    
    #gIsPrint = False
    gIsPrint = True

    model_name = "gpt2/model/gpt2-medium"
    #model_name = "gpt2-medium"

    gTestdata = ["I am Robert, how are you?",
                "I am a student",
                "Hello, this is Eric",
                "How do you od",
                "Hi",
                "How to learn AI",
                "How to learn math",
                "Are you good at math",
                "I am Kevin Liu, how about you?",
                "I do not like you"]

    model,tokenizer = load_model(model_name)

    show_memory("The model is loaded, Main Process")

    batch_test(model,tokenizer,gTestdata,2,64,"warm up",gIsPrint)
 
    asyncio.run( coroutine_test(model,tokenizer,gTestdata,2,64,"Coroutine",gIsPrint) )

    multithread_test(model,tokenizer,gTestdata,2,64,"Multithread",gIsPrint)

    multiprocess_test(model_name,gTestdata,2,64,'Multiprocess', gIsPrint)
 