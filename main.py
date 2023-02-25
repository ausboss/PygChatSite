from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

app = Flask(__name__)

# ToDo: set your torch

# Peepys torch
# load the pre-trained model and tokenizer
# revision = "dev"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = AutoModelForCausalLM.from_pretrained(
#     "PygmalionAI/pygmalion-6b", revision=revision, torch_dtype=torch.float16
# ).to(device)
# tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-6b")


# AusBoss torch
# load the pre-trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "PygmalionAI/pygmalion-6b"
model = torch.load("E:\\PygDiscordBot\\torch-dumps\\pygmalion-6b_dev.pt")
tokenizer = AutoTokenizer.from_pretrained(model_name)



# set up chatbot prompt
chatbot_prompt = "Let's chat! Say something to start the conversation."


# set up chatbot prompt
chatbot_prompt = "Let's chat! Say something to start the conversation."


class Chatbot:
    def __init__(self, char_filename):
        # read character data from JSON file
        with open(char_filename, "r") as f:
            data = json.load(f)
            self.char_name = data["char_name"]
            self.char_persona = data["char_persona"]
            self.char_greeting = data["char_greeting"]
            self.world_scenario = data["world_scenario"]
            self.example_dialogue = data["example_dialogue"]

        # initialize conversation history and character information
        self.conversation_history = f"<START>\n{self.char_name}: {self.char_greeting}\n"
        self.character_info = f"{self.char_name}'s Persona: {self.char_persona}\nScenario: {self.world_scenario}\n"
        self.num_lines_to_keep = 20

    def generate_response(self, input_ids):
        attention_mask = torch.ones_like(input_ids)
        output = model.generate(
            input_ids,
            max_length=2048,
            do_sample=True,
            use_cache=True,
            min_new_tokens=10,
            temperature=0.95,
            repetition_penalty=1.03,
            top_p=0.9,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            )
        if output is not None:
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text
        else:
            return "This is an empty message. Something went wrong. Please check your code!"



    def save_conversation(self, message):
        # add user response to conversation history
        print(f"message: {message}")
        self.conversation_history += f'You: {message}\n'
        print(f'self.conversation_history: {self.conversation_history}')
        # format prompt
        prompt = {
            "prompt": self.character_info + '\n'.join(
                self.conversation_history.split('\n')[-self.num_lines_to_keep:]) + f'{self.char_name}:',
        }

        input_ids = tokenizer.encode(prompt["prompt"], return_tensors="pt").to(device)
        results = self.generate_response(input_ids)
        text_lines = [line.strip() for line in str(results).split("\n")]
        print(text_lines)
        bot_line = next((line for line in reversed(text_lines) if self.char_name in line), None)
        if bot_line is not None:
            bot_line = bot_line.replace(f'{self.char_name}:', '').strip()
            for i in range(len(text_lines)):
                text_lines[i] = text_lines[i].replace(f'{self.char_name}:', '')
        response_text = bot_line
        self.conversation_history += f'{self.char_name}: {response_text}\n'
        return response_text




# initialize chatbot
chatbot = Chatbot('static/tensorsama.json')


#Todo: fix user input not being added to chat history/prompt

# define route for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    print(request.form)  # Debugging statement
    user_input = request.form['input']
    # Process the user's input and generate a chatbot response
    chatbot_response = chatbot.save_conversation(user_input)
    # Return the chatbot response as JSON
    return jsonify({'output': chatbot_response})


# initialize chatbot before first request
@app.before_first_request
def init_chatbot():
    global chatbot
    chatbot = Chatbot('static/tensorsama.json')

# define route for the homepage 
# Todo: make chatbot_prompt show up as first message
@app.route('/')
def home():
    return render_template('home.html', chatbot_prompt=chatbot_prompt)

if __name__ == '__main__':
    app.run(debug=True)