## Web Pyg Project Starting files
Uses flask and local transformers. This is a major WIP and just something I slapped together in an hour with the help of chatGPT


![image](https://i.imgur.com/Yeaie2jl.png)


```python
revision = "dev"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(
     "PygmalionAI/pygmalion-6b", revision=revision, torch_dtype=torch.float16
).to(device)
tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "PygmalionAI/pygmalion-6b"
model = torch.load("E:\\PygDiscordBot\\torch-dumps\\pygmalion-6b_dev.pt")
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

